//
// Free function to construct the cosmic-muon stop stand: an aluminium target
// plus scintillator paddles, all placed in one mother volume (normally
// HallAir) just upstream of the extracted (garage-position) tracker.
// Built only when hasCosmicStopStand = true (see Mu2eWorld.cc).
//
// The presets (centre, tilt, paddle offsets) are computed outside Offline
// (analysis repo, CosmicStopStudy/fullsim/standgeom.py) and passed in as
// SimpleConfig keys. Lengths in mm, angles in degrees.
//
//   string         cosmicStopStand.motherVolume           name of the mother volume
//   int            cosmicStopStand.verbosityLevel         default 0
//   bool           cosmicStopStand.doSurfaceCheck         default false (OR g4.doSurfaceCheck)
//   bool           cosmicStopStand.visible                default true
//   bool           cosmicStopStand.solid                  default false
//   string         cosmicStopStand.target.material
//   vector<double> cosmicStopStand.target.halfLengths     3 values, target frame
//   vector<double> cosmicStopStand.target.centerInMu2e    3 values, Mu2e frame
//   double         cosmicStopStand.tiltDeg                rotation about +x
//   string         cosmicStopStand.paddle.material
//   vector<string> cosmicStopStand.paddle.roles           n entries: top, bottom or side
//   vector<double> cosmicStopStand.paddle.halfLengths     3n values, target frame
//   vector<double> cosmicStopStand.paddle.offsetsInTarget 3n values, paddle centre minus
//                                                         target centre, target frame
//   double         cosmicStopStand.minClearanceToTracker  min gap to the TrackerMother
//                                                         upstream face
//
// Rotation convention: R is the ACTIVE rotation by +tiltDeg about +x (target
// frame -> Mu2e frame); it turns the target top-face normal +y toward +z
// (downstream). The target and every paddle share R, and paddle i is centred
// at target.centerInMu2e + R * offsetsInTarget[i]. G4PVPlacement (via
// nestBox/finishNesting) takes the FRAME rotation, so R.inverse() is passed.
//
// Clearance guard: the most downstream corner of any stand volume must be at
// least minClearanceToTracker upstream of the TrackerMother upstream face
// (tolerance 1e-3 mm, see guardTolerance), otherwise a cet::exception("GEOM")
// is thrown.
//
// Volume names: CosmicStopBlock (target) and CosmicStopPaddle<i> (i = index
// in roles). Sensitive-detector, region and analysis code depend on them; a
// sensitive volume's name is its StepPointMC instance name, which art allows
// to contain only letters and digits.
//
// The mother must not be rotated with respect to Mu2e coordinates.
//

// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Framework includes
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "cetlib_except/exception.h"

// CLHEP includes
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Vector/RotationX.h"
#include "CLHEP/Vector/ThreeVector.h"

// Mu2e includes
#include "Offline/Mu2eG4/inc/constructCosmicStopStand.hh"
#include "Offline/Mu2eG4/inc/MaterialFinder.hh"
#include "Offline/Mu2eG4/inc/nestBox.hh"
#include "Offline/Mu2eG4Helper/inc/AntiLeakRegistry.hh"
#include "Offline/Mu2eG4Helper/inc/Mu2eG4Helper.hh"
#include "Offline/Mu2eG4Helper/inc/VolumeInfo.hh"
#include "Offline/ConfigTools/inc/SimpleConfig.hh"

// G4 includes
#include "Geant4/G4Color.hh"
#include "Geant4/G4Material.hh"
#include "Geant4/G4RotationMatrix.hh"
#include "Geant4/G4ThreeVector.hh"
#include "Geant4/G4VSolid.hh"

using namespace std;

namespace mu2e {

  namespace {

    // One box of the stand: its centre in Mu2e, its half lengths in the
    // target frame and the Mu2e bounding box of the rotated box.
    struct StandBox {
      string         name;
      string         role;
      vector<double> half;
      G4ThreeVector  center;
      G4ThreeVector  lo;
      G4ThreeVector  hi;
    };

    // Read a vector<double> key and require exactly nRequired entries.
    vector<double> getVectorOfSize( SimpleConfig const& config,
                                    string const& key,
                                    size_t nRequired ){
      vector<double> v;
      config.getVectorDouble(key, v);
      if ( v.size() != nRequired ) {
        throw cet::exception("CONFIG")
          << "constructCosmicStopStand: key " << key << " has " << v.size()
          << " entries, expected " << nRequired << "\n";
      }
      return v;
    }

    // Fill the Mu2e bounding box of a box with centre b.center, half lengths
    // b.half (target frame) and active rotation R: the min and max over the
    // 8 corners center + R*(+-hx, +-hy, +-hz).
    void fillBoundingBox( StandBox& b, CLHEP::HepRotationX const& R ){
      bool first = true;
      for ( int sx=-1; sx<=1; sx+=2 ) {
        for ( int sy=-1; sy<=1; sy+=2 ) {
          for ( int sz=-1; sz<=1; sz+=2 ) {
            G4ThreeVector const corner = b.center
              + R*G4ThreeVector( sx*b.half[0], sy*b.half[1], sz*b.half[2] );
            if ( first ) {
              b.lo  = corner;
              b.hi  = corner;
              first = false;
            } else {
              b.lo.set( min(b.lo.x(), corner.x()), min(b.lo.y(), corner.y()), min(b.lo.z(), corner.z()) );
              b.hi.set( max(b.hi.x(), corner.x()), max(b.hi.y(), corner.y()), max(b.hi.z(), corner.z()) );
            }
          }
        }
      }
    }

  }  // end anonymous namespace

  void constructCosmicStopStand( VolumeInfo   const& mother,
                                 SimpleConfig const& config ){

    MaterialFinder materialFinder(config);

    bool const doSurfaceCheck      = config.getBool("g4.doSurfaceCheck",false)
                                  || config.getBool("cosmicStopStand.doSurfaceCheck",false);
    bool const forceAuxEdgeVisible = config.getBool("g4.forceAuxEdgeVisible",false);
    bool const isVisible           = config.getBool("cosmicStopStand.visible",true);
    bool const isSolid             = config.getBool("cosmicStopStand.solid",false);
    int  const verbosity           = config.getInt("cosmicStopStand.verbosityLevel",0);

    // Lengths are printed in fixed notation with 4 decimals (0.1 um) via local
    // ostringstreams, so cout's own format state is never changed.
    int const printDecimals = 4;

    if ( verbosity > 0 ) {
      G4ThreeVector pMin, pMax;
      mother.solid->BoundingLimits(pMin, pMax);
      ostringstream os;
      os << fixed << setprecision(printDecimals);
      os << "constructCosmicStopStand: mother " << mother.name
         << " extent in Mu2e " << mother.centerInMu2e() + pMin
         << " to " << mother.centerInMu2e() + pMax << " mm\n";
      cout << os.str() << flush;
    }

    // ---- Read and validate the configuration.

    vector<double> const targetHalf   = getVectorOfSize(config, "cosmicStopStand.target.halfLengths",  3);
    vector<double> const targetCenter = getVectorOfSize(config, "cosmicStopStand.target.centerInMu2e", 3);
    double const tiltDeg              = config.getDouble("cosmicStopStand.tiltDeg");
    double const minClearance         = config.getDouble("cosmicStopStand.minClearanceToTracker");

    vector<string> roles;
    config.getVectorString("cosmicStopStand.paddle.roles", roles);
    size_t const nPaddles = roles.size();
    for ( auto const& role : roles ) {
      if ( role != "top" && role != "bottom" && role != "side" ) {
        throw cet::exception("CONFIG")
          << "constructCosmicStopStand: key cosmicStopStand.paddle.roles has unknown role '"
          << role << "' (allowed: top, bottom, side)\n";
      }
    }
    vector<double> const padHalf   = getVectorOfSize(config, "cosmicStopStand.paddle.halfLengths",     3*nPaddles);
    vector<double> const padOffset = getVectorOfSize(config, "cosmicStopStand.paddle.offsetsInTarget", 3*nPaddles);

    G4Material* targetMaterial = materialFinder.get("cosmicStopStand.target.material");
    G4Material* paddleMaterial = materialFinder.get("cosmicStopStand.paddle.material");

    // ---- Active rotation (target frame -> Mu2e) and the volumes in Mu2e.

    CLHEP::HepRotationX const R(tiltDeg*CLHEP::deg);
    G4ThreeVector const targetPos(targetCenter[0], targetCenter[1], targetCenter[2]);

    vector<StandBox> boxes;
    boxes.reserve(1+nPaddles);

    StandBox target;
    target.name   = "CosmicStopBlock";
    target.role   = "target";
    target.half   = targetHalf;
    target.center = targetPos;
    fillBoundingBox(target, R);
    boxes.push_back(target);

    for ( size_t i=0; i<nPaddles; ++i ) {
      StandBox p;
      p.name   = "CosmicStopPaddle" + to_string(i);
      p.role   = roles[i];
      p.half   = vector<double>( padHalf.begin()+3*i, padHalf.begin()+3*i+3 );
      p.center = targetPos
        + R*G4ThreeVector( padOffset[3*i], padOffset[3*i+1], padOffset[3*i+2] );
      fillBoundingBox(p, R);
      boxes.push_back(p);
    }

    // ---- Clearance guard against the TrackerMother upstream face.

    Mu2eG4Helper& helper = *(art::ServiceHandle<Mu2eG4Helper>());
    VolumeInfo const& trackerMother = helper.locateVolInfo("TrackerMother");
    G4ThreeVector tmMin, tmMax;
    trackerMother.solid->BoundingLimits(tmMin, tmMax);
    double const trackerFaceZ = trackerMother.centerInMu2e().z() + tmMin.z();

    size_t iMax = 0;
    for ( size_t i=1; i<boxes.size(); ++i ) {
      if ( boxes[i].hi.z() > boxes[iMax].hi.z() ) iMax = i;
    }
    double const zmaxStand = boxes[iMax].hi.z();

    // Presets are written with 6 decimals and the TrackerMother centre is a
    // computed value, so allow 1 um; a real misplacement is far larger.
    double const guardTolerance = 1e-3; // mm

    if ( verbosity > 0 ) {
      ostringstream os;
      os << fixed << setprecision(printDecimals);
      os << "constructCosmicStopStand: tilt " << tiltDeg << " deg about +x, "
         << nPaddles << " paddles, mother " << mother.name << "\n";
      for ( auto const& b : boxes ) {
        os << "constructCosmicStopStand: " << b.name << " (" << b.role
           << ") center in Mu2e " << b.center
           << " mm, Mu2e bounding box " << b.lo << " to " << b.hi << " mm\n";
      }
      os << "constructCosmicStopStand: TrackerMother upstream face z = " << trackerFaceZ
         << " mm, stand max z = " << zmaxStand << " mm (" << boxes[iMax].name
         << "), gap = " << trackerFaceZ - zmaxStand
         << " mm, required >= " << minClearance << " mm (tolerance "
         << guardTolerance << " mm)\n";
      cout << os.str() << flush;
    }

    if ( zmaxStand > trackerFaceZ - minClearance + guardTolerance ) {
      ostringstream os;
      os << fixed << setprecision(printDecimals);
      os << "constructCosmicStopStand: stand reaches z = " << zmaxStand << " mm ("
         << boxes[iMax].name << "), TrackerMother upstream face is at z = " << trackerFaceZ
         << " mm; gap " << trackerFaceZ - zmaxStand
         << " mm is below cosmicStopStand.minClearanceToTracker = " << minClearance
         << " mm (tolerance " << guardTolerance << " mm)\n";
      throw cet::exception("GEOM") << os.str();
    }

    // ---- Placement. One shared frame rotation (inverse of R), none at zero tilt.

    AntiLeakRegistry& reg = helper.antiLeakRegistry();
    G4RotationMatrix const* rot = ( tiltDeg == 0. )
      ? nullptr
      : reg.add(G4RotationMatrix(R.inverse()));

    for ( size_t i=0; i<boxes.size(); ++i ) {
      bool const isTarget = ( i == 0 );
      nestBox( boxes[i].name,
               boxes[i].half,
               isTarget ? targetMaterial : paddleMaterial,
               rot,
               boxes[i].center - mother.centerInMu2e(),
               mother,
               0,
               isVisible,
               isTarget ? G4Color::Gray() : G4Color::Cyan(),
               isSolid,
               forceAuxEdgeVisible,
               true,
               doSurfaceCheck
               );
    }

    if ( verbosity > 0 ) {
      cout << "constructCosmicStopStand: placed " << boxes.size() << " volumes, target material "
           << targetMaterial->GetName() << ", paddle material "
           << paddleMaterial->GetName() << endl;
    }
  }

}  // end namespace mu2e
