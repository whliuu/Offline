//
// Truth dumper for the cosmic-muon stop stand full simulation
// (analysis repo: CosmicStopStudy/docs/2026-10-20-full-sim-design.md, section 4.5;
// the tree contract is Task 3 of CosmicStopStudy/docs/2026-10-20-full-sim-plan.md).
//
// Cosmic muons stop in an Al target (volume CosmicStopBlock) tagged by
// scintillator paddles (CosmicStopPaddle_<i>), upstream of the extracted
// (garage-position) tracker and calorimeter, field off. For each event the
// module turns the Geant4 truth of g4Tag (StepPointMCs, SimParticles) into
// compact records:
//
//   - paddle records  (paddle,  paddleBinNs bin, origin): earliest t, summed E
//   - straw records   (straw,   trkBinNs bin,    origin): earliest t, summed E,
//                                                         E-weighted position
//   - crystal records (crystal, caloBinNs bin,   origin): same as straws
//   - CRV records     (bar,     crvBinNs bin)           : earliest t, summed E
//   - generator truth (GenParticles of genTag)
//   - stops: muons whose end volume is targetName, each with its decay electron
//
// Origin of a record: "stop" = index of the stop the depositing particle
// descends from (SimParticle parent chain), -1 if none; "fromEle" = 1 if it
// descends from that stop's decay electron. Record times are global G4 times
// [ns], energies [MeV], positions [mm] in Mu2e coordinates.
//
// Frames of the StepPointMC positions read here:
//   - paddles (Mu2eG4SensitiveDetector), calorimeter (CaloCrystalSD) and CRV
//     (CRVSD) write  world - mu2eOriginInWorld, i.e. Mu2e coordinates;
//   - tracker (StrawSD) writes  world - DetectorSystem::toMu2e(0), i.e. the
//     tracker (detector system) frame, the frame of Straw::getMidPoint().
// Straw records and straw midpoints are therefore both mapped to Mu2e with
// the same shift, DetectorSystem::toMu2e(0) - mu2eOriginInWorld (see
// trackerToMu2eShift()); all other positions are used as they are.
//
// Output, in the TFileService directory named after the module label:
//   TTree events    one entry per kept event (parameter keep)
//   TTree subruns   one entry per subrun: event counters, CosmicLivetime
//   TTree meta      one entry per job: metaJson and the cosmicStopStand keys
//   TTree geometry  one entry per job: straws, crystals, CRV bars, stand
//                   volumes, Mu2e origin in the G4 world, garage z offset;
//                   all positions in Mu2e coordinates where Mu2eG4 places
//                   the volumes (for the extracted geometry the garage
//                   position is already in the tracker/calo keys, so
//                   garage_z_offset is information, not a shift to apply)
//   TH1D  hTrkPlanes, TH1D hCaloE, TH2D hTrkCalo: every event with good G4
//         status, kept or not (overflow is put in the last bin)
//
// Events with StatusG4::status() > 9 are counted in n_g4bad and skipped.
//
// The geometry tree is filled in beginRun when WorldG4 is already there (in
// a Mu2eG4 job the g4run beginRun on the trigger path usually runs first),
// else on the first event, else in endJob; if WorldG4 never exists (no
// Mu2eG4 in the job) mu2e_origin_in_world is written as three NaNs and the
// tracker shift uses mu2eOriginInWorld = 0.
//

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Core/ModuleMacros.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"
#include "art/Framework/Principal/SubRun.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Vector/RotationX.h"
#include "CLHEP/Vector/ThreeVector.h"

#include "Offline/CalorimeterGeom/inc/Calorimeter.hh"
#include "Offline/ConfigTools/inc/SimpleConfig.hh"
#include "Offline/CosmicRayShieldGeom/inc/CosmicRayShield.hh"
#include "Offline/CosmicRayShieldGeom/inc/CRSScintillatorBar.hh"
#include "Offline/DataProducts/inc/PDGCode.hh"
#include "Offline/DataProducts/inc/StrawId.hh"
#include "Offline/GeometryService/inc/DetectorSystem.hh"
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "Offline/GeometryService/inc/GeometryService.hh"
#include "Offline/GeometryService/inc/WorldG4.hh"
#include "Offline/MCDataProducts/inc/CosmicLivetime.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/PhysicalVolumeInfoMultiCollection.hh"
#include "Offline/MCDataProducts/inc/ProcessCode.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StatusG4.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"
#include "Offline/Mu2eUtilities/inc/PhysicalVolumeMultiHelper.hh"
#include "Offline/TrackerGeom/inc/Tracker.hh"

#include "TH1D.h"
#include "TH2D.h"
#include "TTree.h"

namespace mu2e {

  class CosmicStopStandAna : public art::EDAnalyzer {
  public:
    struct Config {
      using Name    = fhicl::Name;
      using Comment = fhicl::Comment;
      fhicl::Atom<art::InputTag> g4Tag{ Name("g4Tag"),
          Comment("Mu2eG4 module label: StepPointMCs, SimParticles, StatusG4, volume info"),
          art::InputTag("g4run") };
      fhicl::Atom<art::InputTag> genTag{ Name("genTag"), Comment("GenParticleCollection"),
          art::InputTag("generate") };
      fhicl::Atom<art::InputTag> livetimeTag{ Name("livetimeTag"),
          Comment("SubRun CosmicLivetime (CRY); livetime_s = -1 when absent"),
          art::InputTag("generate") };
      fhicl::Atom<std::string> keep{ Name("keep"),
          Comment("Events written to the events tree: all, stopOrPaddle, paddle, detector, detectorOrPaddle") };
      fhicl::Atom<double> paddleBinNs{ Name("paddleBinNs"), Comment("Paddle record time bin [ns]"), 5. };
      fhicl::Atom<double> trkBinNs{ Name("trkBinNs"), Comment("Straw record time bin [ns]"), 50. };
      fhicl::Atom<double> caloBinNs{ Name("caloBinNs"), Comment("Crystal record time bin [ns]"), 50. };
      fhicl::Atom<double> crvBinNs{ Name("crvBinNs"), Comment("CRV bar record time bin [ns]"), 50. };
      fhicl::Atom<double> minEleMomentum{ Name("minEleMomentum"),
          Comment("Decay electron minimum |p| [MeV/c]"), 1. };
      fhicl::Atom<std::string> targetName{ Name("targetName"),
          Comment("End volume that defines a stop"), "CosmicStopBlock" };
      fhicl::Atom<std::string> metaJson{ Name("metaJson"),
          Comment("Job description, copied to the meta tree"), "{}" };
    };
    using Parameters = art::EDAnalyzer::Table<Config>;

    explicit CosmicStopStandAna(Parameters const& conf);

    void beginRun(art::Run const& run) override;
    void beginSubRun(art::SubRun const& sr) override;
    void endSubRun(art::SubRun const& sr) override;
    void analyze(art::Event const& event) override;
    void endJob() override;

  private:
    enum class Keep { all, stopOrPaddle, paddle, detector, detectorOrPaddle };
    static Keep parseKeep(std::string const& s);

    // Which stop a particle descends from, and whether via its decay electron.
    struct Origin { int stop = -1; int fromEle = 0; };

    // One aggregated record: earliest time, summed energy, sum of E * position.
    struct Record {
      double t = 0.;
      double e = 0.;
      CLHEP::Hep3Vector ePos;
    };
    // (detector index, time bin, origin stop, fromEle) -> record
    using RecordKey = std::tuple<int,long,int,int>;
    using RecordMap = std::map<RecordKey, Record>;
    static void addRecord(RecordMap& m, int det, StepPointMC const& s, double binNs, Origin const& o);

    // Branch buffers of the events tree.
    struct EventCols {
      int run = 0, subrun = 0, event = 0;
      int tagPdg = 0, tagIsStop = -1;
      float tagPTop = -1.f;   // tag |p| [MeV/c] at its earliest top-paddle step, -1 if none
      std::string tagEndVol, tagEndProc;
      std::vector<int>   padIdx, padOrigin, padFromEle;
      std::vector<float> padT, padE;
      std::vector<int>   trkPlane, trkPanel, trkStraw, trkOrigin, trkFromEle;
      std::vector<float> trkT, trkE, trkX, trkY, trkZ;
      std::vector<int>   caloCrystal, caloDisk, caloOrigin, caloFromEle;
      std::vector<float> caloT, caloE, caloX, caloY, caloZ;
      std::vector<int>   crvBar;
      std::vector<float> crvT, crvE;
      std::vector<int>   genPdg;
      std::vector<float> genP, genX, genY, genZ, genUx, genUy, genUz;
      std::vector<int>   stopPdg;
      std::vector<float> stopT, stopTStart, stopPEnd, stopX, stopY, stopZ;
      std::vector<std::string> stopProc, stopStartVol;
      std::vector<int>   eleFound;
      std::vector<float> eleP, eleCosZ;
      std::vector<std::string> eleProc;
      void clear();
    };

    // Branch buffers of the subruns tree.
    struct SubrunCols {
      int run = 0, subrun = 0, nEvents = 0, nKept = 0, nG4Bad = 0;
      double livetime = -1.;
    };

    // Branch buffers of the meta tree (the cosmicStopStand.* keys).
    struct MetaCols {
      std::string json;
      std::vector<std::string> roles;
      std::vector<double> targetHalf, targetCenter, paddleHalf, paddleOffset;
      double tiltDeg = 0.;
    };

    // Branch buffers of the geometry tree.
    struct GeomCols {
      std::vector<int>    strawPlane, strawPanel, strawStraw;
      std::vector<double> strawMidX, strawMidY, strawMidZ, strawDirX, strawDirY, strawDirZ, strawHalfLen;
      std::vector<int>    crystalId, crystalDisk;
      std::vector<double> crystalX, crystalY, crystalZ, crystalHxy, crystalHz;
      std::vector<int>    crvBar;
      std::vector<double> crvX, crvY, crvZ, crvHx, crvHy, crvHz;
      std::vector<std::string> volName;
      std::vector<double> volX, volY, volZ, volHx, volHy, volHz;
      double tiltDeg = 0.;
      std::vector<double> mu2eOriginInWorld;
      double garageZOffset = 0.;
    };

    void readStandConfig();
    CLHEP::Hep3Vector const& trackerToMu2eShift();
    void fillGeometry();
    void addStandVolume(std::string const& name, CLHEP::Hep3Vector const& center, double const* half);

    Origin origin(art::Ptr<SimParticle> p);
    static bool isDecayOrCapture(ProcessCode const& cc);
    void findStops(SimParticleCollection const& sims, PhysicalVolumeMultiHelper const& vi);
    void paddleRecords(art::Event const& event, PhysicalVolumeMultiHelper const& vi, RecordMap& padMap);
    void detectorRecords(art::Event const& event, RecordMap& trkMap, RecordMap& caloMap,
                         RecordMap& crvMap, std::set<int>& planes, double& caloE);
    bool keepEvent(bool hasStop, bool hasPaddle, bool hasDetector) const;
    void flushRecords(RecordMap const& padMap, RecordMap const& trkMap,
                      RecordMap const& caloMap, RecordMap const& crvMap);
    void fillGenTruth(art::Event const& event);
    void clearEvent();

    Config conf_;
    Keep   keep_;

    std::vector<art::InputTag> paddleTags_;   // g4Tag label, instance CosmicStopPaddle_<i>
    bool runSeen_    = false;
    bool geomFilled_ = false;

    // Tracker (detector system) frame -> Mu2e, set on first use (trackerToMu2eShift()).
    CLHEP::Hep3Vector trkShift_;
    bool trkShiftSet_ = false;

    // Per-event lookup tables.
    std::unordered_map<std::size_t,int>    stopIndex_;   // muon key -> stop index
    std::unordered_map<std::size_t,int>    eleIndex_;    // electron key -> stop index
    std::unordered_map<std::size_t,Origin> originCache_;

    EventCols  ev_;
    SubrunCols sr_;
    MetaCols   meta_;
    GeomCols   geo_;

    TTree* eventsTree_   = nullptr;
    TTree* subrunsTree_  = nullptr;
    TTree* metaTree_     = nullptr;
    TTree* geometryTree_ = nullptr;
    TH1D*  hTrkPlanes_   = nullptr;
    TH1D*  hCaloE_       = nullptr;
    TH2D*  hTrkCalo_     = nullptr;
  };

  //================================================================
  CosmicStopStandAna::CosmicStopStandAna(Parameters const& conf)
    : art::EDAnalyzer(conf), conf_(conf()), keep_(parseKeep(conf().keep())) {

    for ( double const b : { conf_.paddleBinNs(), conf_.trkBinNs(), conf_.caloBinNs(), conf_.crvBinNs() } ) {
      if ( !(b > 0.) ) {
        throw cet::exception("CONFIG") << "CosmicStopStandAna: record time bins must be > 0 ns, got " << b << "\n";
      }
    }

    art::ServiceHandle<art::TFileService> tfs;

    eventsTree_ = tfs->make<TTree>("events", "Cosmic stop stand: records, generator truth and stops");
    eventsTree_->Branch("run",          &ev_.run);
    eventsTree_->Branch("subrun",       &ev_.subrun);
    eventsTree_->Branch("event",        &ev_.event);
    eventsTree_->Branch("tag_pdg",      &ev_.tagPdg);
    eventsTree_->Branch("tag_endVol",   &ev_.tagEndVol);
    eventsTree_->Branch("tag_endProc",  &ev_.tagEndProc);
    eventsTree_->Branch("tag_isStop",   &ev_.tagIsStop);
    eventsTree_->Branch("tag_pTop",     &ev_.tagPTop);
    eventsTree_->Branch("pad_idx",      &ev_.padIdx);
    eventsTree_->Branch("pad_t",        &ev_.padT);
    eventsTree_->Branch("pad_e",        &ev_.padE);
    eventsTree_->Branch("pad_origin",   &ev_.padOrigin);
    eventsTree_->Branch("pad_fromEle",  &ev_.padFromEle);
    eventsTree_->Branch("trk_plane",    &ev_.trkPlane);
    eventsTree_->Branch("trk_panel",    &ev_.trkPanel);
    eventsTree_->Branch("trk_straw",    &ev_.trkStraw);
    eventsTree_->Branch("trk_t",        &ev_.trkT);
    eventsTree_->Branch("trk_e",        &ev_.trkE);
    eventsTree_->Branch("trk_x",        &ev_.trkX);
    eventsTree_->Branch("trk_y",        &ev_.trkY);
    eventsTree_->Branch("trk_z",        &ev_.trkZ);
    eventsTree_->Branch("trk_origin",   &ev_.trkOrigin);
    eventsTree_->Branch("trk_fromEle",  &ev_.trkFromEle);
    eventsTree_->Branch("calo_crystal", &ev_.caloCrystal);
    eventsTree_->Branch("calo_disk",    &ev_.caloDisk);
    eventsTree_->Branch("calo_t",       &ev_.caloT);
    eventsTree_->Branch("calo_e",       &ev_.caloE);
    eventsTree_->Branch("calo_x",       &ev_.caloX);
    eventsTree_->Branch("calo_y",       &ev_.caloY);
    eventsTree_->Branch("calo_z",       &ev_.caloZ);
    eventsTree_->Branch("calo_origin",  &ev_.caloOrigin);
    eventsTree_->Branch("calo_fromEle", &ev_.caloFromEle);
    eventsTree_->Branch("crv_bar",      &ev_.crvBar);
    eventsTree_->Branch("crv_t",        &ev_.crvT);
    eventsTree_->Branch("crv_e",        &ev_.crvE);
    eventsTree_->Branch("gen_pdg",      &ev_.genPdg);
    eventsTree_->Branch("gen_p",        &ev_.genP);
    eventsTree_->Branch("gen_x",        &ev_.genX);
    eventsTree_->Branch("gen_y",        &ev_.genY);
    eventsTree_->Branch("gen_z",        &ev_.genZ);
    eventsTree_->Branch("gen_ux",       &ev_.genUx);
    eventsTree_->Branch("gen_uy",       &ev_.genUy);
    eventsTree_->Branch("gen_uz",       &ev_.genUz);
    eventsTree_->Branch("stop_pdg",     &ev_.stopPdg);
    eventsTree_->Branch("stop_t",       &ev_.stopT);
    eventsTree_->Branch("stop_tStart",  &ev_.stopTStart);
    eventsTree_->Branch("stop_pEnd",    &ev_.stopPEnd);
    eventsTree_->Branch("stop_x",       &ev_.stopX);
    eventsTree_->Branch("stop_y",       &ev_.stopY);
    eventsTree_->Branch("stop_z",       &ev_.stopZ);
    eventsTree_->Branch("stop_proc",    &ev_.stopProc);
    eventsTree_->Branch("stop_startVol",&ev_.stopStartVol);
    eventsTree_->Branch("ele_found",    &ev_.eleFound);
    eventsTree_->Branch("ele_p",        &ev_.eleP);
    eventsTree_->Branch("ele_cosz",     &ev_.eleCosZ);
    eventsTree_->Branch("ele_proc",     &ev_.eleProc);

    subrunsTree_ = tfs->make<TTree>("subruns", "Per-subrun event counters and livetime");
    subrunsTree_->Branch("run",        &sr_.run);
    subrunsTree_->Branch("subrun",     &sr_.subrun);
    subrunsTree_->Branch("n_events",   &sr_.nEvents);
    subrunsTree_->Branch("n_kept",     &sr_.nKept);
    subrunsTree_->Branch("n_g4bad",    &sr_.nG4Bad);
    subrunsTree_->Branch("livetime_s", &sr_.livetime);

    metaTree_ = tfs->make<TTree>("meta", "Job description and stand keys");
    metaTree_->Branch("json",          &meta_.json);
    metaTree_->Branch("roles",         &meta_.roles);
    metaTree_->Branch("target_half",   &meta_.targetHalf);
    metaTree_->Branch("target_center", &meta_.targetCenter);
    metaTree_->Branch("paddle_half",   &meta_.paddleHalf);
    metaTree_->Branch("paddle_offset", &meta_.paddleOffset);
    metaTree_->Branch("tilt_deg",      &meta_.tiltDeg);

    geometryTree_ = tfs->make<TTree>("geometry", "Straws, crystals, CRV bars and stand volumes in Mu2e coordinates");
    geometryTree_->Branch("straw_plane",          &geo_.strawPlane);
    geometryTree_->Branch("straw_panel",          &geo_.strawPanel);
    geometryTree_->Branch("straw_straw",          &geo_.strawStraw);
    geometryTree_->Branch("straw_mid_x",          &geo_.strawMidX);
    geometryTree_->Branch("straw_mid_y",          &geo_.strawMidY);
    geometryTree_->Branch("straw_mid_z",          &geo_.strawMidZ);
    geometryTree_->Branch("straw_dir_x",          &geo_.strawDirX);
    geometryTree_->Branch("straw_dir_y",          &geo_.strawDirY);
    geometryTree_->Branch("straw_dir_z",          &geo_.strawDirZ);
    geometryTree_->Branch("straw_halflen",        &geo_.strawHalfLen);
    geometryTree_->Branch("crystal_id",           &geo_.crystalId);
    geometryTree_->Branch("crystal_disk",         &geo_.crystalDisk);
    geometryTree_->Branch("crystal_x",            &geo_.crystalX);
    geometryTree_->Branch("crystal_y",            &geo_.crystalY);
    geometryTree_->Branch("crystal_z",            &geo_.crystalZ);
    geometryTree_->Branch("crystal_hxy",          &geo_.crystalHxy);
    geometryTree_->Branch("crystal_hz",           &geo_.crystalHz);
    geometryTree_->Branch("crv_bar",              &geo_.crvBar);
    geometryTree_->Branch("crv_x",                &geo_.crvX);
    geometryTree_->Branch("crv_y",                &geo_.crvY);
    geometryTree_->Branch("crv_z",                &geo_.crvZ);
    geometryTree_->Branch("crv_hx",               &geo_.crvHx);
    geometryTree_->Branch("crv_hy",               &geo_.crvHy);
    geometryTree_->Branch("crv_hz",               &geo_.crvHz);
    geometryTree_->Branch("vol_name",             &geo_.volName);
    geometryTree_->Branch("vol_x",                &geo_.volX);
    geometryTree_->Branch("vol_y",                &geo_.volY);
    geometryTree_->Branch("vol_z",                &geo_.volZ);
    geometryTree_->Branch("vol_hx",               &geo_.volHx);
    geometryTree_->Branch("vol_hy",               &geo_.volHy);
    geometryTree_->Branch("vol_hz",               &geo_.volHz);
    geometryTree_->Branch("tilt_deg",             &geo_.tiltDeg);
    geometryTree_->Branch("mu2e_origin_in_world", &geo_.mu2eOriginInWorld);
    geometryTree_->Branch("garage_z_offset",      &geo_.garageZOffset);

    hTrkPlanes_ = tfs->make<TH1D>("hTrkPlanes",
        "Distinct straw planes with energy, every good event;planes;events", 38, -0.5, 37.5);
    hCaloE_     = tfs->make<TH1D>("hCaloE",
        "Total crystal energy, every good event;E [MeV];events", 1000, 0., 1000.);
    hTrkCalo_   = tfs->make<TH2D>("hTrkCalo",
        "Planes vs total crystal energy, every good event;planes;E [MeV]", 38, -0.5, 37.5, 1000, 0., 1000.);
  }

  //================================================================
  CosmicStopStandAna::Keep CosmicStopStandAna::parseKeep(std::string const& s) {
    if ( s == "all" )              return Keep::all;
    if ( s == "stopOrPaddle" )     return Keep::stopOrPaddle;
    if ( s == "paddle" )           return Keep::paddle;
    if ( s == "detector" )         return Keep::detector;
    if ( s == "detectorOrPaddle" ) return Keep::detectorOrPaddle;
    throw cet::exception("CONFIG") << "CosmicStopStandAna: keep = '" << s
                                   << "', must be one of all, stopOrPaddle, paddle, detector, detectorOrPaddle\n";
  }

  //================================================================
  // Run, subrun and job transitions.

  void CosmicStopStandAna::beginRun(art::Run const&) {
    if ( !runSeen_ ) {
      readStandConfig();
      metaTree_->Fill();
      runSeen_ = true;
    }
    if ( !geomFilled_ && art::ServiceHandle<GeometryService>()->hasElement<WorldG4>() ) fillGeometry();
  }

  void CosmicStopStandAna::beginSubRun(art::SubRun const&) {
    sr_ = SubrunCols();
  }

  void CosmicStopStandAna::endSubRun(art::SubRun const& sr) {
    sr_.run      = static_cast<int>(sr.run());
    sr_.subrun   = static_cast<int>(sr.subRun());
    sr_.livetime = -1.;
    auto const lh = sr.getHandle<CosmicLivetime>(conf_.livetimeTag());
    if ( lh.isValid() ) sr_.livetime = lh->liveTime();
    subrunsTree_->Fill();
  }

  void CosmicStopStandAna::endJob() {
    // Only for a job that saw a run but no event and no WorldG4 at beginRun.
    if ( runSeen_ && !geomFilled_ ) fillGeometry();
  }

  //================================================================
  // Stand keys (meta tree) and the geometry tree.

  void CosmicStopStandAna::readStandConfig() {
    SimpleConfig const& c = art::ServiceHandle<GeometryService>()->config();
    meta_.json = conf_.metaJson();
    c.getVectorString("cosmicStopStand.paddle.roles", meta_.roles);
    int const n3 = 3*static_cast<int>(meta_.roles.size());
    c.getVectorDouble("cosmicStopStand.target.halfLengths",     meta_.targetHalf,   3);
    c.getVectorDouble("cosmicStopStand.target.centerInMu2e",    meta_.targetCenter, 3);
    c.getVectorDouble("cosmicStopStand.paddle.halfLengths",     meta_.paddleHalf,   n3);
    c.getVectorDouble("cosmicStopStand.paddle.offsetsInTarget", meta_.paddleOffset, n3);
    meta_.tiltDeg = c.getDouble("cosmicStopStand.tiltDeg");

    paddleTags_.clear();
    for ( std::size_t i=0; i<meta_.roles.size(); ++i ) {
      paddleTags_.emplace_back(conf_.g4Tag().label(), "CosmicStopPaddle_" + std::to_string(i));
    }
  }

  // Shift that maps a tracker-frame position to Mu2e: StrawSD stores
  //   p = world - DetectorSystem::toMu2e(0)      (StrawSD.cc, detectorOrigin)
  // and the Mu2e frame of the other SDs is  world - mu2eOriginInWorld, so
  //   Mu2e = p + DetectorSystem::toMu2e(0) - mu2eOriginInWorld.
  // StrawSD's own check (verbosity > 3) states that the G4 straw centre is at
  // world = getMidPoint() + DetectorSystem::toMu2e(0), so the same shift maps the
  // straw midpoints. Set once per job; mu2eOriginInWorld = 0 if WorldG4 is absent.
  CLHEP::Hep3Vector const& CosmicStopStandAna::trackerToMu2eShift() {
    if ( !trkShiftSet_ ) {
      art::ServiceHandle<GeometryService> geom;
      trkShift_ = GeomHandle<DetectorSystem>()->toMu2e(CLHEP::Hep3Vector());
      if ( geom->hasElement<WorldG4>() ) {
        trkShift_ -= GeomHandle<WorldG4>()->mu2eOriginInWorld();
      } else {
        mf::LogWarning("CosmicStopStandAna") << "WorldG4 not available: tracker frame -> Mu2e "
                                             << "shift assumes mu2eOriginInWorld = 0\n";
      }
      trkShiftSet_ = true;
    }
    return trkShift_;
  }

  void CosmicStopStandAna::addStandVolume(std::string const& name, CLHEP::Hep3Vector const& center,
                                          double const* half) {
    geo_.volName.push_back(name);
    geo_.volX.push_back(center.x());
    geo_.volY.push_back(center.y());
    geo_.volZ.push_back(center.z());
    geo_.volHx.push_back(half[0]);
    geo_.volHy.push_back(half[1]);
    geo_.volHz.push_back(half[2]);
  }

  void CosmicStopStandAna::fillGeometry() {
    art::ServiceHandle<GeometryService> geom;
    SimpleConfig const& c = geom->config();

    // Garage position. garage.zOffset (constructDS.cc) moves only the DS
    // volumes (garageFakeDS3Vacuum). The extracted geometry moves the tracker
    // and calorimeter with their own keys (geom_common_extracted_v04.txt:
    // mu2e.detectorSystemZ0, tracker.z0, tracker.mother.z0 = 24171/24175,
    // calorimeter.caloMotherZ0/Z1), and Mu2eG4 places TrackerMother and
    // CalorimeterMother at those Mu2e positions inside the shifted DS3 volume.
    // So the geometry objects below are already where Mu2eG4 puts them and no
    // shift is added; garage_z_offset is recorded for information only.
    geo_.garageZOffset = c.getBool("inGaragePosition", false) ? c.getDouble("garage.zOffset") : 0.;

    // Straws: every straw of every existing plane (Mu2eG4 skips the others).
    // Straw::getMidPoint() is in the tracker frame; the same shift as for the
    // straw records maps it to Mu2e (trackerToMu2eShift()).
    if ( geom->hasElement<Tracker>() ) {
      Tracker const& tracker = *GeomHandle<Tracker>();
      CLHEP::Hep3Vector const shift = trackerToMu2eShift();
      for ( Straw const& straw : tracker.straws() ) {
        StrawId const& id = straw.id();
        if ( !id.valid() || straw.halfLength() <= 0. || !tracker.planeExists(id) ) continue;
        CLHEP::Hep3Vector const mid = straw.getMidPoint() + shift;
        CLHEP::Hep3Vector const& dir = straw.getDirection();
        geo_.strawPlane.push_back(id.plane());
        geo_.strawPanel.push_back(id.panel());
        geo_.strawStraw.push_back(id.straw());
        geo_.strawMidX.push_back(mid.x());
        geo_.strawMidY.push_back(mid.y());
        geo_.strawMidZ.push_back(mid.z());
        geo_.strawDirX.push_back(dir.x());
        geo_.strawDirY.push_back(dir.y());
        geo_.strawDirZ.push_back(dir.z());
        geo_.strawHalfLen.push_back(straw.halfLength());
      }
    }

    // Crystals: Crystal::localPosition() is the centre of the crystal front face
    // in the disk frame (DiskCalorimeterMaker: diskOriginToCrystalOrigin), with
    // the crystal along +z of the disk; the centre is half a crystal length
    // further, mapped to Mu2e by the disk-to-Mu2e transform.
    if ( geom->hasElement<Calorimeter>() ) {
      Calorimeter const& cal = *GeomHandle<Calorimeter>();
      double const hxy = 0.5*cal.G4Info().get<double>("crystalXYLength");
      double const hz  = 0.5*cal.G4Info().get<double>("crystalZLength");
      for ( unsigned i=0; i<cal.nCrystals(); ++i ) {
        Crystal const& cr = cal.crystal(i);
        CLHEP::Hep3Vector const ctr =
          cal.diskToMu2e(cr.diskID(), cr.localPosition() + CLHEP::Hep3Vector(0., 0., hz));
        geo_.crystalId.push_back(static_cast<int>(cr.ID()));
        geo_.crystalDisk.push_back(static_cast<int>(cr.diskID()));
        geo_.crystalX.push_back(ctr.x());
        geo_.crystalY.push_back(ctr.y());
        geo_.crystalZ.push_back(ctr.z());
        geo_.crystalHxy.push_back(hxy);
        geo_.crystalHz.push_back(hz);
      }
    }

    // CRV bars: centre in Mu2e and half lengths along Mu2e x, y, z (the bars
    // are placed unrotated, CosmicRayShieldMaker/constructCRV).
    if ( geom->hasElement<CosmicRayShield>() ) {
      CosmicRayShield const& crs = *GeomHandle<CosmicRayShield>();
      for ( auto const& bp : crs.getAllCRSScintillatorBars() ) {
        CRSScintillatorBar const& bar = *bp;
        CLHEP::Hep3Vector const& pos = bar.getPosition();
        std::vector<double> const& half = bar.getHalfLengths();
        geo_.crvBar.push_back(bar.index().asInt());
        geo_.crvX.push_back(pos.x());
        geo_.crvY.push_back(pos.y());
        geo_.crvZ.push_back(pos.z());
        geo_.crvHx.push_back(half.at(0));
        geo_.crvHy.push_back(half.at(1));
        geo_.crvHz.push_back(half.at(2));
      }
    }

    // Stand volumes, as placed by constructCosmicStopStand.cc: R is the active
    // rotation by +tiltDeg about +x (target frame -> Mu2e), paddle i is centred
    // at target centre + R * offset_i. Half lengths are in the target frame.
    CLHEP::HepRotationX const R(meta_.tiltDeg*CLHEP::deg);
    CLHEP::Hep3Vector const c0(meta_.targetCenter[0], meta_.targetCenter[1], meta_.targetCenter[2]);
    addStandVolume("CosmicStopBlock", c0, &meta_.targetHalf[0]);
    for ( std::size_t i=0; i<meta_.roles.size(); ++i ) {
      CLHEP::Hep3Vector const off(meta_.paddleOffset[3*i], meta_.paddleOffset[3*i+1], meta_.paddleOffset[3*i+2]);
      addStandVolume("CosmicStopPaddle_" + std::to_string(i), c0 + R*off, &meta_.paddleHalf[3*i]);
    }
    geo_.tiltDeg = meta_.tiltDeg;

    if ( geom->hasElement<WorldG4>() ) {
      CLHEP::Hep3Vector const& o = GeomHandle<WorldG4>()->mu2eOriginInWorld();
      geo_.mu2eOriginInWorld = { o.x(), o.y(), o.z() };
    } else {
      double const nan = std::numeric_limits<double>::quiet_NaN();
      geo_.mu2eOriginInWorld = { nan, nan, nan };
      mf::LogWarning("CosmicStopStandAna") << "WorldG4 not available (no Mu2eG4 in this job?): "
                                           << "mu2e_origin_in_world written as NaN\n";
    }

    geometryTree_->Fill();
    geomFilled_ = true;
  }

  //================================================================
  // Records and origins.

  CosmicStopStandAna::Origin CosmicStopStandAna::origin(art::Ptr<SimParticle> p) {
    std::size_t const k0 = p.key();
    auto const cached = originCache_.find(k0);
    if ( cached != originCache_.end() ) return cached->second;
    Origin o;
    while ( p.isNonnull() ) {
      auto const ie = eleIndex_.find(p.key());
      if ( ie != eleIndex_.end() ) { o.stop = ie->second; o.fromEle = 1; break; }
      art::Ptr<SimParticle> const parent = p->parent();
      if ( parent.isNonnull() ) {
        auto const is = stopIndex_.find(parent.key());
        if ( is != stopIndex_.end() ) { o.stop = is->second; break; }
      }
      p = parent;
    }
    originCache_[k0] = o;
    return o;
  }

  void CosmicStopStandAna::addRecord(RecordMap& m, int det, StepPointMC const& s, double binNs, Origin const& o) {
    double const t = s.time();
    double const e = s.totalEDep();
    long const bin = static_cast<long>(std::floor(t / binNs));
    auto const [it, inserted] = m.try_emplace(RecordKey(det, bin, o.stop, o.fromEle));
    Record& r = it->second;
    if ( inserted || t < r.t ) r.t = t;
    r.e    += e;
    r.ePos += e*s.position();
  }

  // Creation codes accepted for the decay electron: the old set (Decay,
  // DecayWithSpin, DIO, muMinusCaptureAtRest) widened to every code whose
  // name contains "Decay".
  bool CosmicStopStandAna::isDecayOrCapture(ProcessCode const& cc) {
    if ( cc == ProcessCode::DIO || cc == ProcessCode::muMinusCaptureAtRest ) return true;
    return cc.name().find("Decay") != std::string::npos;
  }

  // Muons that end in targetName, and their decay electrons.
  void CosmicStopStandAna::findStops(SimParticleCollection const& sims, PhysicalVolumeMultiHelper const& vi) {
    for ( auto const& [key, sim] : sims ) {
      if ( std::abs(sim.pdgId()) != PDGCode::mu_minus ) continue;
      if ( !sim.endDefined() ) continue;
      if ( vi.endVolume(sim).name() != conf_.targetName() ) continue;
      int const k = static_cast<int>(ev_.stopPdg.size());
      stopIndex_[key.asUint()] = k;
      ev_.stopPdg.push_back(sim.pdgId());
      // Mu2eG4 records the times of a muon decaying at rest from before the decay
      // (Mu2eG4UserHelpers.cc), so use the latest daughter start time as the
      // decay or capture time; fall back to the end time if there are no daughters.
      double tDecay = sim.endGlobalTime();
      if ( !sim.daughters().empty() ) {
        tDecay = sim.daughters().front()->startGlobalTime();
        for ( auto const& d : sim.daughters() ) tDecay = std::max<double>(tDecay, d->startGlobalTime());
      }
      ev_.stopT.push_back(tDecay);
      ev_.stopTStart.push_back(sim.startGlobalTime());
      ev_.stopPEnd.push_back(sim.endMomentum().vect().mag());
      ev_.stopX.push_back(sim.endPosition().x());
      ev_.stopY.push_back(sim.endPosition().y());
      ev_.stopZ.push_back(sim.endPosition().z());
      ev_.stopProc.push_back(sim.stoppingCode().name());
      ev_.stopStartVol.push_back(vi.startVolume(sim).name());

      // Decay electron: the highest-|p| direct daughter of the right charge
      // (mu- -> e-, mu+ -> e+) from a decay or capture process, |p| >= minEleMomentum.
      PDGCode::type const eleType = ( sim.pdgId() > 0 ) ? PDGCode::e_minus : PDGCode::e_plus;
      art::Ptr<SimParticle> best;
      for ( auto const& d : sim.daughters() ) {
        if ( d->pdgId() != eleType ) continue;
        if ( !isDecayOrCapture(d->creationCode()) ) continue;
        double const p = d->startMomentum().vect().mag();
        if ( p < conf_.minEleMomentum() ) continue;
        if ( best.isNull() || p > best->startMomentum().vect().mag() ) best = d;
      }
      ev_.eleFound.push_back(best.isNonnull() ? 1 : 0);
      if ( best.isNonnull() ) {
        eleIndex_[best.key()] = k;
        ev_.eleP.push_back(best->startMomentum().vect().mag());
        ev_.eleCosZ.push_back(best->startMomentum().vect().cosTheta());
        ev_.eleProc.push_back(best->creationCode().name());
      } else {
        ev_.eleP.push_back(-1.f);
        ev_.eleCosZ.push_back(-2.f);
        ev_.eleProc.push_back("");
      }
    }
  }

  // Paddle records; tagging particle = the muon with the largest top-paddle eDep,
  // else the particle with the largest top-paddle eDep, else the largest eDep in
  // any paddle.
  void CosmicStopStandAna::paddleRecords(art::Event const& event, PhysicalVolumeMultiHelper const& vi,
                                         RecordMap& padMap) {
    std::unordered_map<std::size_t, std::pair<double, art::Ptr<SimParticle>>> eTop, eAny;
    std::unordered_map<std::size_t, std::pair<double, double>> pTop;   // key -> (earliest top step time, |p| there)
    for ( std::size_t i=0; i<paddleTags_.size(); ++i ) {
      for ( auto const& s : *event.getValidHandle<StepPointMCCollection>(paddleTags_[i]) ) {
        if ( s.totalEDep() <= 0. ) continue;
        addRecord(padMap, static_cast<int>(i), s, conf_.paddleBinNs(), origin(s.simParticle()));
        auto& a = eAny[s.simParticle().key()];
        a.first += s.totalEDep(); a.second = s.simParticle();
        if ( meta_.roles[i] == "top" ) {
          auto& b = eTop[s.simParticle().key()];
          b.first += s.totalEDep(); b.second = s.simParticle();
          auto const ip = pTop.find(s.simParticle().key());
          if ( ip == pTop.end() || s.time() < ip->second.first ) {
            pTop[s.simParticle().key()] = { s.time(), s.momentum().mag() };
          }
        }
      }
    }
    auto pick = [](auto const& m, bool muonOnly) {
      art::Ptr<SimParticle> tp;
      double eMax = -1.;
      for ( auto const& kv : m ) {
        if ( muonOnly && std::abs(kv.second.second->pdgId()) != PDGCode::mu_minus ) continue;
        if ( kv.second.first > eMax ) { eMax = kv.second.first; tp = kv.second.second; }
      }
      return tp;
    };
    art::Ptr<SimParticle> tp = pick(eTop, true);
    if ( tp.isNull() ) tp = pick(eTop, false);
    if ( tp.isNull() ) tp = pick(eAny, false);
    if ( tp.isNonnull() ) {
      ev_.tagPdg     = tp->pdgId();
      ev_.tagEndVol  = vi.endVolume(*tp).name();
      ev_.tagEndProc = tp->stoppingCode().name();
      auto const it = stopIndex_.find(tp.key());
      ev_.tagIsStop  = ( it == stopIndex_.end() ) ? -1 : it->second;
      auto const ip = pTop.find(tp.key());
      if ( ip != pTop.end() ) ev_.tagPTop = static_cast<float>(ip->second.second);
    }
  }

  // Straw, crystal and CRV records for all particles, plus the inputs of the
  // all-event histograms (distinct planes with energy, total crystal energy).
  void CosmicStopStandAna::detectorRecords(art::Event const& event, RecordMap& trkMap, RecordMap& caloMap,
                                           RecordMap& crvMap, std::set<int>& planes, double& caloE) {
    std::string const& g4 = conf_.g4Tag().label();

    for ( auto const& s : *event.getValidHandle<StepPointMCCollection>(art::InputTag(g4, "tracker")) ) {
      if ( s.totalEDep() <= 0. ) continue;
      StrawId const sid = s.strawId();
      planes.insert(sid.plane());
      addRecord(trkMap, sid.asUint16(), s, conf_.trkBinNs(), origin(s.simParticle()));
    }

    auto const& caloSteps = *event.getValidHandle<StepPointMCCollection>(art::InputTag(g4, "calorimeter"));
    if ( !caloSteps.empty() ) {
      GeomHandle<Calorimeter> cal;
      for ( auto const& s : caloSteps ) {
        if ( s.totalEDep() <= 0. ) continue;
        int const id = static_cast<int>(cal->crystal(s.volumeId()).ID());
        caloE += s.totalEDep();
        addRecord(caloMap, id, s, conf_.caloBinNs(), origin(s.simParticle()));
      }
    }

    for ( auto const& s : *event.getValidHandle<StepPointMCCollection>(art::InputTag(g4, "CRV")) ) {
      if ( s.totalEDep() <= 0. ) continue;
      addRecord(crvMap, s.barIndex().asInt(), s, conf_.crvBinNs(), Origin());
    }
  }

  bool CosmicStopStandAna::keepEvent(bool hasStop, bool hasPaddle, bool hasDetector) const {
    switch ( keep_ ) {
    case Keep::all:              return true;
    case Keep::stopOrPaddle:     return hasStop || hasPaddle;
    case Keep::paddle:           return hasPaddle;
    case Keep::detector:         return hasDetector;
    case Keep::detectorOrPaddle: return hasDetector || hasPaddle;
    }
    return true;
  }

  void CosmicStopStandAna::flushRecords(RecordMap const& padMap, RecordMap const& trkMap,
                                        RecordMap const& caloMap, RecordMap const& crvMap) {
    for ( auto const& [key, r] : padMap ) {
      ev_.padIdx.push_back(std::get<0>(key));
      ev_.padOrigin.push_back(std::get<2>(key));
      ev_.padFromEle.push_back(std::get<3>(key));
      ev_.padT.push_back(r.t);
      ev_.padE.push_back(r.e);
    }

    // Straw records: StrawSD positions are in the tracker frame; map the
    // energy-weighted mean to Mu2e (a translation commutes with the mean).
    CLHEP::Hep3Vector const trkShift = trkMap.empty() ? CLHEP::Hep3Vector() : trackerToMu2eShift();
    for ( auto const& [key, r] : trkMap ) {
      StrawId const sid(static_cast<uint16_t>(std::get<0>(key)));
      CLHEP::Hep3Vector const x = r.ePos/r.e + trkShift;
      ev_.trkPlane.push_back(sid.plane());
      ev_.trkPanel.push_back(sid.panel());
      ev_.trkStraw.push_back(sid.straw());
      ev_.trkOrigin.push_back(std::get<2>(key));
      ev_.trkFromEle.push_back(std::get<3>(key));
      ev_.trkT.push_back(r.t);
      ev_.trkE.push_back(r.e);
      ev_.trkX.push_back(x.x());
      ev_.trkY.push_back(x.y());
      ev_.trkZ.push_back(x.z());
    }

    if ( !caloMap.empty() ) {
      GeomHandle<Calorimeter> cal;
      for ( auto const& [key, r] : caloMap ) {
        int const id = std::get<0>(key);
        CLHEP::Hep3Vector const x = r.ePos/r.e;
        ev_.caloCrystal.push_back(id);
        ev_.caloDisk.push_back(static_cast<int>(cal->crystal(id).diskID()));
        ev_.caloOrigin.push_back(std::get<2>(key));
        ev_.caloFromEle.push_back(std::get<3>(key));
        ev_.caloT.push_back(r.t);
        ev_.caloE.push_back(r.e);
        ev_.caloX.push_back(x.x());
        ev_.caloY.push_back(x.y());
        ev_.caloZ.push_back(x.z());
      }
    }

    for ( auto const& [key, r] : crvMap ) {
      ev_.crvBar.push_back(std::get<0>(key));
      ev_.crvT.push_back(r.t);
      ev_.crvE.push_back(r.e);
    }
  }

  // Generator truth: every GenParticle, |p| [MeV/c], start point [mm] and
  // unit direction (zero for a particle at rest).
  void CosmicStopStandAna::fillGenTruth(art::Event const& event) {
    for ( auto const& g : *event.getValidHandle<GenParticleCollection>(conf_.genTag()) ) {
      CLHEP::Hep3Vector const p = g.momentum().vect();
      CLHEP::Hep3Vector const u = p.unit();
      ev_.genPdg.push_back(g.pdgId());
      ev_.genP.push_back(p.mag());
      ev_.genX.push_back(g.position().x());
      ev_.genY.push_back(g.position().y());
      ev_.genZ.push_back(g.position().z());
      ev_.genUx.push_back(u.x());
      ev_.genUy.push_back(u.y());
      ev_.genUz.push_back(u.z());
    }
  }

  void CosmicStopStandAna::EventCols::clear() {
    run = subrun = event = 0;
    tagPdg = 0; tagIsStop = -1; tagPTop = -1.f; tagEndVol.clear(); tagEndProc.clear();
    for ( auto* v : { &padIdx, &padOrigin, &padFromEle, &trkPlane, &trkPanel, &trkStraw, &trkOrigin,
                      &trkFromEle, &caloCrystal, &caloDisk, &caloOrigin, &caloFromEle, &crvBar, &genPdg,
                      &stopPdg, &eleFound } ) v->clear();
    for ( auto* v : { &padT, &padE, &trkT, &trkE, &trkX, &trkY, &trkZ, &caloT, &caloE, &caloX, &caloY,
                      &caloZ, &crvT, &crvE, &genP, &genX, &genY, &genZ, &genUx, &genUy, &genUz, &stopT,
                      &stopTStart, &stopPEnd, &stopX, &stopY, &stopZ, &eleP, &eleCosZ } ) v->clear();
    for ( auto* v : { &stopProc, &stopStartVol, &eleProc } ) v->clear();
  }

  void CosmicStopStandAna::clearEvent() {
    stopIndex_.clear(); eleIndex_.clear(); originCache_.clear();
    ev_.clear();
  }

  //================================================================
  void CosmicStopStandAna::analyze(art::Event const& event) {
    if ( !geomFilled_ ) fillGeometry();
    ++sr_.nEvents;

    auto const& g4Status = *event.getValidHandle<StatusG4>(conf_.g4Tag());
    if ( g4Status.status() > 9 ) { ++sr_.nG4Bad; return; }

    clearEvent();
    ev_.run    = static_cast<int>(event.run());
    ev_.subrun = static_cast<int>(event.subRun());
    ev_.event  = static_cast<int>(event.event());

    auto const& pvh = event.getSubRun().getValidHandle<PhysicalVolumeInfoMultiCollection>(conf_.g4Tag());
    PhysicalVolumeMultiHelper const vi(&*pvh);
    auto const& sims = *event.getValidHandle<SimParticleCollection>(conf_.g4Tag());

    // 1) Stops first: the record origins refer to them.
    findStops(sims, vi);

    // 2) Paddle, straw, crystal and CRV records.
    RecordMap padMap, trkMap, caloMap, crvMap;
    paddleRecords(event, vi, padMap);
    std::set<int> planes;
    double caloE = 0.;
    detectorRecords(event, trkMap, caloMap, crvMap, planes, caloE);

    // 3) All-event histograms (every good event, kept or not); overflow -> last bin.
    int    const nPlanes = std::min(static_cast<int>(planes.size()), 37);
    double const eCalo   = std::min(caloE, hCaloE_->GetXaxis()->GetBinCenter(hCaloE_->GetNbinsX()));
    hTrkPlanes_->Fill(nPlanes);
    hCaloE_->Fill(eCalo);
    hTrkCalo_->Fill(nPlanes, eCalo);

    // 4) Keep decision, then the events tree.
    bool const hasStop     = !ev_.stopPdg.empty();
    bool const hasPaddle   = !padMap.empty();
    bool const hasDetector = !trkMap.empty() || !caloMap.empty();
    if ( !keepEvent(hasStop, hasPaddle, hasDetector) ) return;

    flushRecords(padMap, trkMap, caloMap, crvMap);
    fillGenTruth(event);
    eventsTree_->Fill();
    ++sr_.nKept;
  }

}  // end namespace mu2e

DEFINE_ART_MODULE(mu2e::CosmicStopStandAna)
