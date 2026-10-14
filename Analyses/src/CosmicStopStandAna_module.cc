//
// Truth dumper for the cosmic-muon stop stand study
// (analysis/CosmicStopStudy/docs/2026-10-14-cosmic-stop-stand-design.md).
//
// For each event with a paddle step or a muon ending in the stand block it
// writes hit records for the paddles, tracker and calorimeter, each tagged
// with its origin (which block stop it descends from, and whether it comes
// from that stop's decay electron), plus truth for each block stop.
// Every event fills the all-event histograms; every subrun fills the subrun
// tree with the cosmic livetime when the generator provides one.
//

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

#include "Offline/CalorimeterGeom/inc/Calorimeter.hh"
#include "Offline/ConfigTools/inc/SimpleConfig.hh"
#include "Offline/DataProducts/inc/PDGCode.hh"
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "Offline/GeometryService/inc/GeometryService.hh"
#include "Offline/MCDataProducts/inc/CosmicLivetime.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/PhysicalVolumeInfoMultiCollection.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"
#include "Offline/Mu2eUtilities/inc/PhysicalVolumeMultiHelper.hh"

#include "TH1D.h"
#include "TTree.h"

namespace mu2e {

  class CosmicStopStandAna : public art::EDAnalyzer {
  public:
    struct Config {
      using Name    = fhicl::Name;
      using Comment = fhicl::Comment;
      fhicl::Atom<art::InputTag> g4Tag{ Name("g4Tag"), Comment("Mu2eG4 module label"), art::InputTag("g4run") };
      fhicl::Atom<art::InputTag> genTag{ Name("genTag"), Comment("GenParticles for genOnly mode"), art::InputTag("generate") };
      fhicl::Atom<art::InputTag> livetimeTag{ Name("livetimeTag"), Comment("SubRun CosmicLivetime"), art::InputTag("generate") };
      fhicl::Atom<bool>   genOnly{ Name("genOnly"), Comment("Only count generated muons crossing the top paddle plane"), false };
      fhicl::Atom<double> paddleBinNs{ Name("paddleBinNs"), Comment("Paddle record time bin [ns]"), 5. };
      fhicl::Atom<double> trkBinNs{ Name("trkBinNs"), Comment("Tracker record time bin [ns]"), 50. };
      fhicl::Atom<double> caloBinNs{ Name("caloBinNs"), Comment("Calo record time bin [ns]"), 50. };
      fhicl::Atom<double> minEleMomentum{ Name("minEleMomentum"), Comment("Decay electron minimum |p| [MeV/c]"), 1. };
    };
    using Parameters = art::EDAnalyzer::Table<Config>;

    explicit CosmicStopStandAna(Parameters const& conf);

    void beginRun(art::Run const& run) override;
    void beginSubRun(art::SubRun const& sr) override;
    void endSubRun(art::SubRun const& sr) override;
    void analyze(art::Event const& event) override;

  private:
    struct Origin { int stop = -1; int fromEle = 0; };
    // (detector index, time bin, origin stop, fromEle) -> (earliest time, summed energy)
    using RecordMap = std::map<std::tuple<int,long,int,int>, std::pair<float,float>>;

    Origin origin(art::Ptr<SimParticle> p);
    static void addRecord(RecordMap& m, int det, double t, double e, double binNs, Origin const& o);
    static void flush(RecordMap const& m, std::vector<int>& det, std::vector<float>& t,
                      std::vector<float>& e, std::vector<int>& org, std::vector<int>& fromEle);
    void countGenCrossings(art::Event const& event);
    void clearEvent();

    Config conf_;
    std::string const blockName_ = "CosmicStopBlock";

    // Stand geometry, read from the GeometryService config in beginRun.
    std::vector<std::string> roles_;
    std::vector<double> blockHalf_, blockCenter_, padHalf_, padCenter_;
    double topY_ = 0., topXmin_ = 0., topXmax_ = 0., topZmin_ = 0., topZmax_ = 0.;
    bool metaFilled_ = false;

    // Per-event lookup tables.
    std::unordered_map<std::size_t,int>    stopIndex_;   // muon key -> stop index
    std::unordered_map<std::size_t,int>    eleIndex_;    // electron key -> stop index
    std::unordered_map<std::size_t,Origin> originCache_;

    // Subrun counters.
    unsigned long long nEvents_ = 0, nGenCross_ = 0;
    double livetime_ = -1.;

    TTree* stops_   = nullptr;
    TTree* subruns_ = nullptr;
    TTree* meta_    = nullptr;
    TH1D*  hAllTrkPlanes_ = nullptr;
    TH1D*  hAllCaloE_     = nullptr;

    unsigned run_ = 0, subrun_ = 0, event_ = 0;
    std::vector<int>   padIdx_, padOrigin_, padFromEle_;
    std::vector<float> padT_, padE_;
    std::vector<int>   trkPlane_, trkOrigin_, trkFromEle_;
    std::vector<float> trkT_, trkE_;
    std::vector<int>   caloDisk_, caloOrigin_, caloFromEle_;
    std::vector<float> caloT_, caloE_;
    int tagPdg_ = 0, tagIsStop_ = -1;
    std::string tagEndVol_, tagEndProc_;
    std::vector<int>   stopPdg_, eleFound_;
    std::vector<float> stopT_, stopTStart_, stopPEnd_, stopX_, stopY_, stopZ_, eleP_, eleCosZ_;
    std::vector<std::string> stopProc_, eleProc_;
    std::string blockNameBranch_ = "CosmicStopBlock";
  };

  CosmicStopStandAna::CosmicStopStandAna(Parameters const& conf)
    : art::EDAnalyzer(conf), conf_(conf()) {

    art::ServiceHandle<art::TFileService> tfs;

    stops_ = tfs->make<TTree>("stops", "Cosmic stop stand: hit records and block stops");
    stops_->Branch("run", &run_);
    stops_->Branch("subrun", &subrun_);
    stops_->Branch("event", &event_);
    stops_->Branch("pad_idx", &padIdx_);
    stops_->Branch("pad_t", &padT_);
    stops_->Branch("pad_e", &padE_);
    stops_->Branch("pad_origin", &padOrigin_);
    stops_->Branch("pad_fromEle", &padFromEle_);
    stops_->Branch("trk_plane", &trkPlane_);
    stops_->Branch("trk_t", &trkT_);
    stops_->Branch("trk_e", &trkE_);
    stops_->Branch("trk_origin", &trkOrigin_);
    stops_->Branch("trk_fromEle", &trkFromEle_);
    stops_->Branch("calo_disk", &caloDisk_);
    stops_->Branch("calo_t", &caloT_);
    stops_->Branch("calo_e", &caloE_);
    stops_->Branch("calo_origin", &caloOrigin_);
    stops_->Branch("calo_fromEle", &caloFromEle_);
    stops_->Branch("tag_pdg", &tagPdg_);
    stops_->Branch("tag_endVol", &tagEndVol_);
    stops_->Branch("tag_endProc", &tagEndProc_);
    stops_->Branch("tag_isStop", &tagIsStop_);
    stops_->Branch("stop_pdg", &stopPdg_);
    stops_->Branch("stop_t", &stopT_);
    stops_->Branch("stop_tStart", &stopTStart_);
    stops_->Branch("stop_pEnd", &stopPEnd_);
    stops_->Branch("stop_x", &stopX_);
    stops_->Branch("stop_y", &stopY_);
    stops_->Branch("stop_z", &stopZ_);
    stops_->Branch("stop_proc", &stopProc_);
    stops_->Branch("ele_found", &eleFound_);
    stops_->Branch("ele_p", &eleP_);
    stops_->Branch("ele_cosz", &eleCosZ_);
    stops_->Branch("ele_proc", &eleProc_);

    subruns_ = tfs->make<TTree>("subruns", "Per-subrun livetime and counters");
    subruns_->Branch("run", &run_);
    subruns_->Branch("subrun", &subrun_);
    subruns_->Branch("livetime_s", &livetime_);
    subruns_->Branch("n_events", &nEvents_);
    subruns_->Branch("n_gen_cross", &nGenCross_);

    meta_ = tfs->make<TTree>("meta", "Stand geometry");
    meta_->Branch("block_name", &blockNameBranch_);
    meta_->Branch("paddle_roles", &roles_);
    meta_->Branch("block_half", &blockHalf_);
    meta_->Branch("block_center", &blockCenter_);
    meta_->Branch("paddle_half", &padHalf_);
    meta_->Branch("paddle_center", &padCenter_);

    hAllTrkPlanes_ = tfs->make<TH1D>("hAllTrkPlanes", "Distinct straw planes hit, all events;planes;events", 38, -0.5, 37.5);
    hAllCaloE_     = tfs->make<TH1D>("hAllCaloE", "Total calo crystal energy, all events;E [MeV];events", 1000, 0., 1000.);
  }

  void CosmicStopStandAna::beginRun(art::Run const&) {
    if ( metaFilled_ ) return;
    SimpleConfig const& c = art::ServiceHandle<GeometryService>()->config();
    c.getVectorString("cosmicStopStand.paddle.roles", roles_);
    c.getVectorDouble("cosmicStopStand.block.halfLengths",   blockHalf_,   3);
    c.getVectorDouble("cosmicStopStand.block.centerInMu2e",  blockCenter_, 3);
    c.getVectorDouble("cosmicStopStand.paddle.halfLengths",   padHalf_,   3*roles_.size());
    c.getVectorDouble("cosmicStopStand.paddle.centersInMu2e", padCenter_, 3*roles_.size());

    // Top face and x-z footprint of the highest top paddle, for genOnly crossings.
    bool found = false;
    for ( std::size_t i=0; i<roles_.size(); ++i ) {
      if ( roles_[i] != "top" ) continue;
      double const yTop = padCenter_[3*i+1] + padHalf_[3*i+1];
      if ( !found || yTop > topY_ ) {
        found    = true;
        topY_    = yTop;
        topXmin_ = padCenter_[3*i]   - padHalf_[3*i];
        topXmax_ = padCenter_[3*i]   + padHalf_[3*i];
        topZmin_ = padCenter_[3*i+2] - padHalf_[3*i+2];
        topZmax_ = padCenter_[3*i+2] + padHalf_[3*i+2];
      }
    }
    if ( !found ) {
      throw cet::exception("CONFIG") << "CosmicStopStandAna: no paddle with role 'top'\n";
    }
    meta_->Fill();
    metaFilled_ = true;
  }

  void CosmicStopStandAna::beginSubRun(art::SubRun const&) {
    nEvents_   = 0;
    nGenCross_ = 0;
  }

  void CosmicStopStandAna::endSubRun(art::SubRun const& sr) {
    run_      = sr.run();
    subrun_   = sr.subRun();
    livetime_ = -1.;
    auto const lh = sr.getHandle<CosmicLivetime>(conf_.livetimeTag());
    if ( lh.isValid() ) livetime_ = lh->liveTime();
    subruns_->Fill();
  }

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

  void CosmicStopStandAna::addRecord(RecordMap& m, int det, double t, double e, double binNs, Origin const& o) {
    long const bin = static_cast<long>(std::floor(t / binNs));
    auto const key = std::make_tuple(det, bin, o.stop, o.fromEle);
    auto it = m.find(key);
    if ( it == m.end() ) {
      m.emplace(key, std::make_pair(static_cast<float>(t), static_cast<float>(e)));
    } else {
      it->second.first   = std::min(it->second.first, static_cast<float>(t));
      it->second.second += static_cast<float>(e);
    }
  }

  void CosmicStopStandAna::flush(RecordMap const& m, std::vector<int>& det, std::vector<float>& t,
                                 std::vector<float>& e, std::vector<int>& org, std::vector<int>& fromEle) {
    for ( auto const& [key, val] : m ) {
      det.push_back(std::get<0>(key));
      org.push_back(std::get<2>(key));
      fromEle.push_back(std::get<3>(key));
      t.push_back(val.first);
      e.push_back(val.second);
    }
  }

  void CosmicStopStandAna::countGenCrossings(art::Event const& event) {
    auto const& gens = *event.getValidHandle<GenParticleCollection>(conf_.genTag());
    for ( auto const& g : gens ) {
      if ( std::abs(g.pdgId()) != PDGCode::mu_minus ) continue;
      CLHEP::Hep3Vector const& x0 = g.position();
      CLHEP::Hep3Vector const  u  = g.momentum().vect().unit();
      if ( u.y() >= 0. ) continue;
      double const s = (topY_ - x0.y()) / u.y();
      if ( s < 0. ) continue;
      double const x = x0.x() + s*u.x();
      double const z = x0.z() + s*u.z();
      if ( x >= topXmin_ && x <= topXmax_ && z >= topZmin_ && z <= topZmax_ ) ++nGenCross_;
    }
  }

  void CosmicStopStandAna::clearEvent() {
    stopIndex_.clear(); eleIndex_.clear(); originCache_.clear();
    for ( auto* v : { &padIdx_, &padOrigin_, &padFromEle_, &trkPlane_, &trkOrigin_, &trkFromEle_,
                      &caloDisk_, &caloOrigin_, &caloFromEle_, &stopPdg_, &eleFound_ } ) v->clear();
    for ( auto* v : { &padT_, &padE_, &trkT_, &trkE_, &caloT_, &caloE_, &stopT_, &stopTStart_,
                      &stopPEnd_, &stopX_, &stopY_, &stopZ_, &eleP_, &eleCosZ_ } ) v->clear();
    stopProc_.clear(); eleProc_.clear();
    tagPdg_ = 0; tagIsStop_ = -1; tagEndVol_.clear(); tagEndProc_.clear();
  }

  void CosmicStopStandAna::analyze(art::Event const& event) {
    ++nEvents_;
    if ( conf_.genOnly() ) { countGenCrossings(event); return; }

    clearEvent();
    run_ = event.run(); subrun_ = event.subRun(); event_ = event.event();

    auto const& pvh = event.getSubRun().getValidHandle<PhysicalVolumeInfoMultiCollection>(conf_.g4Tag());
    PhysicalVolumeMultiHelper vi(&*pvh);
    auto const& sims = *event.getValidHandle<SimParticleCollection>(conf_.g4Tag());

    // 1) Muons that end in the block, and their decay electrons.
    for ( auto const& [key, sim] : sims ) {
      if ( std::abs(sim.pdgId()) != PDGCode::mu_minus ) continue;
      if ( !sim.endDefined() ) continue;
      if ( vi.endVolume(sim).name() != blockName_ ) continue;
      int const k = static_cast<int>(stopPdg_.size());
      stopIndex_[key.asUint()] = k;
      stopPdg_.push_back(sim.pdgId());
      // Mu2eG4 records the times of a muon decaying at rest from before the decay
      // (Mu2eG4UserHelpers.cc), so use the latest daughter start time as the
      // decay or capture time; fall back to the end time if there are no daughters.
      double tDecay = sim.endGlobalTime();
      if ( !sim.daughters().empty() ) {
        tDecay = sim.daughters().front()->startGlobalTime();
        for ( auto const& d : sim.daughters() ) tDecay = std::max<double>(tDecay, d->startGlobalTime());
      }
      stopT_.push_back(tDecay);
      stopTStart_.push_back(sim.startGlobalTime());
      stopPEnd_.push_back(sim.endMomentum().vect().mag());
      stopX_.push_back(sim.endPosition().x());
      stopY_.push_back(sim.endPosition().y());
      stopZ_.push_back(sim.endPosition().z());
      stopProc_.push_back(sim.stoppingCode().name());

      // Decay electron: the most energetic direct daughter of the right charge (mu- -> e-,
      // mu+ -> e+) created by a decay, DIO or capture process.
      PDGCode::type const eleType = ( sim.pdgId() > 0 ) ? PDGCode::e_minus : PDGCode::e_plus;
      art::Ptr<SimParticle> best;
      for ( auto const& d : sim.daughters() ) {
        if ( d->pdgId() != eleType ) continue;
        ProcessCode const cc = d->creationCode();
        if ( !( cc == ProcessCode::Decay || cc == ProcessCode::DecayWithSpin ||
                cc == ProcessCode::DIO   || cc == ProcessCode::muMinusCaptureAtRest ) ) continue;
        double const p = d->startMomentum().vect().mag();
        if ( p < conf_.minEleMomentum() ) continue;
        if ( best.isNull() || p > best->startMomentum().vect().mag() ) best = d;
      }
      eleFound_.push_back(best.isNonnull() ? 1 : 0);
      if ( best.isNonnull() ) {
        eleIndex_[best.key()] = k;
        eleP_.push_back(best->startMomentum().vect().mag());
        eleCosZ_.push_back(best->startMomentum().vect().cosTheta());
        eleProc_.push_back(best->creationCode().name());
      } else {
        eleP_.push_back(-1.f);
        eleCosZ_.push_back(-2.f);
        eleProc_.push_back("");
      }
    }

    // 2) Paddle records; tagging particle = the muon with the largest top-paddle eDep, else the
    //    particle with the largest top-paddle eDep, else the largest eDep in any paddle.
    RecordMap padMap, trkMap, caloMap;
    std::unordered_map<std::size_t, std::pair<double, art::Ptr<SimParticle>>> eTop, eAny;
    for ( std::size_t i=0; i<roles_.size(); ++i ) {
      art::InputTag const tag(conf_.g4Tag().label(), "CosmicStopPaddle_" + std::to_string(i));
      for ( auto const& s : *event.getValidHandle<StepPointMCCollection>(tag) ) {
        if ( s.totalEDep() <= 0. ) continue;
        addRecord(padMap, static_cast<int>(i), s.time(), s.totalEDep(), conf_.paddleBinNs(), origin(s.simParticle()));
        auto& a = eAny[s.simParticle().key()];
        a.first += s.totalEDep(); a.second = s.simParticle();
        if ( roles_[i] == "top" ) {
          auto& b = eTop[s.simParticle().key()];
          b.first += s.totalEDep(); b.second = s.simParticle();
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
      tagPdg_     = tp->pdgId();
      tagEndVol_  = vi.endVolume(*tp).name();
      tagEndProc_ = tp->stoppingCode().name();
      auto const it = stopIndex_.find(tp.key());
      tagIsStop_  = ( it == stopIndex_.end() ) ? -1 : it->second;
    }

    // 3) Tracker and calo records for all particles; all-event histograms.
    std::set<int> allPlanes;
    art::InputTag const trkTag(conf_.g4Tag().label(), "tracker");
    for ( auto const& s : *event.getValidHandle<StepPointMCCollection>(trkTag) ) {
      if ( s.totalEDep() <= 0. ) continue;
      int const plane = s.strawId().plane();
      allPlanes.insert(plane);
      addRecord(trkMap, plane, s.time(), s.totalEDep(), conf_.trkBinNs(), origin(s.simParticle()));
    }
    GeomHandle<Calorimeter> cal;
    double allCaloE = 0.;
    art::InputTag const caloTag(conf_.g4Tag().label(), "calorimeter");
    for ( auto const& s : *event.getValidHandle<StepPointMCCollection>(caloTag) ) {
      if ( s.totalEDep() <= 0. ) continue;
      int const disk = cal->crystal(s.volumeId()).diskID();
      allCaloE += s.totalEDep();
      addRecord(caloMap, disk, s.time(), s.totalEDep(), conf_.caloBinNs(), origin(s.simParticle()));
    }
    hAllTrkPlanes_->Fill(allPlanes.size());
    hAllCaloE_->Fill(allCaloE);

    if ( padMap.empty() && stopPdg_.empty() ) return;
    flush(padMap,  padIdx_,   padT_,  padE_,  padOrigin_,  padFromEle_);
    flush(trkMap,  trkPlane_, trkT_,  trkE_,  trkOrigin_,  trkFromEle_);
    flush(caloMap, caloDisk_, caloT_, caloE_, caloOrigin_, caloFromEle_);
    stops_->Fill();
  }

}  // end namespace mu2e

DEFINE_ART_MODULE(mu2e::CosmicStopStandAna)
