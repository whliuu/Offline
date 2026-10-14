//
// Free function to construct the cosmic-muon stop stand: an absorber block
// plus scintillator paddles, all placed in one mother volume.
// Keys: cosmicStopStand.* (see Offline/Mu2eG4/geom/cosmicStopStand_v01.txt).
// The mother must not be rotated with respect to Mu2e coordinates.
//

// C++ includes
#include <iostream>
#include <string>
#include <vector>

// Framework includes
#include "cetlib_except/exception.h"

// Mu2e includes
#include "Offline/Mu2eG4/inc/constructCosmicStopStand.hh"
#include "Offline/Mu2eG4/inc/MaterialFinder.hh"
#include "Offline/Mu2eG4/inc/nestBox.hh"
#include "Offline/ConfigTools/inc/SimpleConfig.hh"

// G4 includes
#include "Geant4/G4Color.hh"
#include "Geant4/G4Material.hh"
#include "Geant4/G4ThreeVector.hh"
#include "Geant4/G4VSolid.hh"

using namespace std;

namespace mu2e {

  void constructCosmicStopStand( VolumeInfo   const& mother,
                                 SimpleConfig const& config ){

    MaterialFinder materialFinder(config);

    bool const doSurfaceCheck      = config.getBool("g4.doSurfaceCheck",false)
                                  || config.getBool("cosmicStopStand.doSurfaceCheck",false);
    bool const forceAuxEdgeVisible = config.getBool("g4.forceAuxEdgeVisible",false);
    bool const isVisible           = config.getBool("cosmicStopStand.visible",true);
    bool const isSolid             = config.getBool("cosmicStopStand.solid",false);
    int  const verbosity           = config.getInt("cosmicStopStand.verbosityLevel",0);

    if ( verbosity > 0 ) {
      G4ThreeVector pMin, pMax;
      mother.solid->BoundingLimits(pMin, pMax);
      cout << "constructCosmicStopStand: mother " << mother.name
           << " extent in Mu2e " << mother.centerInMu2e() + pMin
           << " to " << mother.centerInMu2e() + pMax << " mm" << endl;
    }

    // The block.
    vector<double> blockHalf, blockCenter;
    config.getVectorDouble("cosmicStopStand.block.halfLengths",  blockHalf,   3);
    config.getVectorDouble("cosmicStopStand.block.centerInMu2e", blockCenter, 3);
    G4Material* blockMaterial = materialFinder.get("cosmicStopStand.block.material");
    G4ThreeVector const blockPos(blockCenter[0], blockCenter[1], blockCenter[2]);

    nestBox( "CosmicStopBlock",
             blockHalf,
             blockMaterial,
             nullptr,
             blockPos - mother.centerInMu2e(),
             mother,
             0,
             isVisible,
             G4Color::Gray(),
             isSolid,
             forceAuxEdgeVisible,
             true,
             doSurfaceCheck
             );

    if ( verbosity > 0 ) {
      cout << "constructCosmicStopStand: CosmicStopBlock center in Mu2e " << blockPos
           << " mm, material " << blockMaterial->GetName() << endl;
    }

    // The paddles: 3 half lengths and 3 centre coordinates per role entry.
    vector<string> roles;
    vector<double> padHalf, padCenter;
    config.getVectorString("cosmicStopStand.paddle.roles",         roles);
    config.getVectorDouble("cosmicStopStand.paddle.halfLengths",   padHalf);
    config.getVectorDouble("cosmicStopStand.paddle.centersInMu2e", padCenter);

    if ( padHalf.size() != 3*roles.size() || padCenter.size() != 3*roles.size() ) {
      throw cet::exception("GEOM")
        << "constructCosmicStopStand: need 3 halfLengths and 3 centersInMu2e per paddle; got "
        << roles.size() << " roles, " << padHalf.size() << " halfLengths, "
        << padCenter.size() << " centersInMu2e\n";
    }
    for ( auto const& role : roles ) {
      if ( role != "top" && role != "bottom" && role != "side" ) {
        throw cet::exception("GEOM")
          << "constructCosmicStopStand: unknown paddle role '" << role
          << "' (allowed: top, bottom, side)\n";
      }
    }

    G4Material* paddleMaterial = materialFinder.get("cosmicStopStand.paddle.material");

    for ( size_t i=0; i<roles.size(); ++i ) {
      vector<double> const half( padHalf.begin()+3*i, padHalf.begin()+3*i+3 );
      G4ThreeVector const pos( padCenter[3*i], padCenter[3*i+1], padCenter[3*i+2] );
      string const name = "CosmicStopPaddle_" + to_string(i);

      nestBox( name,
               half,
               paddleMaterial,
               nullptr,
               pos - mother.centerInMu2e(),
               mother,
               0,
               isVisible,
               G4Color::Cyan(),
               isSolid,
               forceAuxEdgeVisible,
               true,
               doSurfaceCheck
               );

      if ( verbosity > 0 ) {
        cout << "constructCosmicStopStand: " << name << " (" << roles[i]
             << ") center in Mu2e " << pos << " mm" << endl;
      }
    }
  }

}  // end namespace mu2e
