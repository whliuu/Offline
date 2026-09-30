// Fires the discrete STM photon lines (muonic-Al X-rays and muon-capture
// gammas) at the HPGe as they arrive from the stopping target: a parallel
// beam spread uniformly over the spot-size-collimator aperture, one Poisson
// count per line per microspill, with per-line timing inside the 1695 ns
// microspill and an optional spill (supercycle) structure.
//
// Each art event is one microspill (EmptyEvent source), as in the HPGe
// digitisation chain.
//
// Line timing (per line, FCL "timing"):
//   "prompt"     : at the muon stop time              (66, 347 keV X-rays)
//   "delayed"    : stop time + exponential(lifetime)  (1809 keV capture gamma)
//   "continuous" : uniform in the microspill, and also emitted off-spill
//                  (844 keV from 27Mg, tau = 818 s >> any beam structure)
// The muon stop time within the microspill is Gaussian (stopTimeMean,
// stopTimeSigma) or sampled from a TH1 (stopTimeHistFile/Name, x in ns).
// All times are folded into [0, micropulse): in steady state a photon
// delayed into a later microspill is statistically the same as one from an
// earlier stop landing in this microspill.
//
// Rates: "weight" is each line's relative intensity averaged over whole
// supercycles (what a long run integrates), and meanPhotonsPerMicrospill is
// the matching time-averaged total. Beam-correlated lines (prompt, delayed)
// only exist on-spill, so their on-spill rate is the average / duty factor
// (duty = spillOn / supercycle); continuous lines keep the average rate.
//   spillMode "onSpill"    : every microspill is on-spill (default)
//   spillMode "supercycle" : microspill n is on-spill if
//                            (n * micropulse) mod supercycle < spillOn,
//                            n = firstMicrospill + eventNumber - 1
//   spillMode "flat"       : every line at its time-averaged rate, no spill
//
// Usage (FCL):
//   generate : {
//     module_type : STMLineSourceGun
//     x : -3944.64  y : 0.  z : 40195.   # beam centre, start plane
//     beamRadius  : 3.99                 # mm, SSC HPGe front hole
//     meanPhotonsPerMicrospill : 0.0678  # 40 kHz time-averaged
//     lines : [
//       { energy : 0.34683  weight : 1.0    timing : "prompt" },
//       { energy : 1.80865  weight : 0.6004 timing : "delayed"  lifetime : 864. },
//       { energy : 0.84376  weight : 0.0755 timing : "continuous" }
//     ]
//   }
//
// Author: Leo Liu

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/OptionalAtom.h"
#include "fhiclcpp/types/Sequence.h"
#include "fhiclcpp/types/Table.h"
#include "messagefacility/MessageLogger/MessageLogger.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Random/RandExponential.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"

#include "Offline/DataProducts/inc/PDGCode.hh"
#include "Offline/MCDataProducts/inc/GenId.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/SeedService/inc/SeedService.hh"

#include "TFile.h"
#include "TH1.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TString.h"

namespace mu2e {

  class STMLineSourceGun : public art::EDProducer {
  public:
    using Name    = fhicl::Name;
    using Comment = fhicl::Comment;

    struct LineConfig {
      fhicl::Atom<double> energy {
        Name("energy"), Comment("Line energy [MeV]") };
      fhicl::Atom<double> weight {
        Name("weight"),
        Comment("Relative intensity at the aperture, averaged over whole supercycles (any normalisation)") };
      fhicl::Atom<std::string> timing {
        Name("timing"),
        Comment("\"prompt\" (at the muon stop time), \"delayed\" (stop time + exponential(lifetime)) "
                "or \"continuous\" (uniform in the microspill, also emitted off-spill)") };
      fhicl::OptionalAtom<double> lifetime {
        Name("lifetime"), Comment("Mean lifetime [ns]; required for timing = \"delayed\"") };
    };

    struct Config {
      fhicl::Sequence<fhicl::Table<LineConfig>> lines {
        Name("lines"), Comment("Discrete photon lines") };
      fhicl::Atom<double> meanPhotonsPerMicrospill {
        Name("meanPhotonsPerMicrospill"),
        Comment("Mean photons per microspill, all lines, averaged over whole supercycles") };
      fhicl::Atom<double> x {
        Name("x"), Comment("Beam centre x on the start plane [mm]") };
      fhicl::Atom<double> y {
        Name("y"), Comment("Beam centre y on the start plane [mm]") };
      fhicl::Atom<double> z {
        Name("z"), Comment("Beam centre z on the start plane [mm]") };
      fhicl::Atom<double> beamRadius {
        Name("beamRadius"),
        Comment("Radius of the uniform disk (perpendicular to aim) the photons start on [mm]") };
      fhicl::Atom<double> aimX {
        Name("aimX"), Comment("Beam direction x (unnormalised)"), 0.0 };
      fhicl::Atom<double> aimY {
        Name("aimY"), Comment("Beam direction y (unnormalised)"), 0.0 };
      fhicl::Atom<double> aimZ {
        Name("aimZ"), Comment("Beam direction z (unnormalised)"), 1.0 };
      fhicl::Atom<double> divergence {
        Name("divergence"), Comment("Half-angle of the direction cone around aim [deg]; 0 = parallel"), 0.0 };
      fhicl::Atom<double> stopTimeMean {
        Name("stopTimeMean"), Comment("Muon stop time in the microspill, Gaussian mean [ns]"), 450.0 };
      fhicl::Atom<double> stopTimeSigma {
        Name("stopTimeSigma"), Comment("Muon stop time in the microspill, Gaussian sigma [ns]"), 125.0 };
      fhicl::OptionalAtom<std::string> stopTimeHistFile {
        Name("stopTimeHistFile"), Comment("ROOT file with a stop-time TH1 (x in ns); overrides the Gaussian") };
      fhicl::OptionalAtom<std::string> stopTimeHistName {
        Name("stopTimeHistName"), Comment("Name of the stop-time TH1 in stopTimeHistFile") };
      fhicl::Atom<std::string> spillMode {
        Name("spillMode"), Comment("\"onSpill\", \"supercycle\" or \"flat\""), "onSpill" };
      fhicl::Atom<double> supercycle {
        Name("supercycle"), Comment("Supercycle period [ns]"), 1.33e9 };
      fhicl::Atom<double> spillOn {
        Name("spillOn"), Comment("Beam-on time at the start of each supercycle [ns]"), 4.92e8 };
      fhicl::Atom<double> micropulse {
        Name("micropulse"), Comment("Microspill length [ns]"), 1695.0 };
      fhicl::Atom<unsigned long> firstMicrospill {
        Name("firstMicrospill"), Comment("Microspill index of event number 1 (for split jobs)"), 0 };
      fhicl::Atom<bool> doHistograms {
        Name("doHistograms"), Comment("Book generator-level diagnostic histograms"), false };
    };

    using Parameters = art::EDProducer::Table<Config>;
    explicit STMLineSourceGun(const Parameters& conf);
    void produce(art::Event& event) override;

  private:
    enum class Timing { prompt, delayed, continuous };
    enum class SpillMode { onSpill, supercycle, flat };

    struct Line {
      double energy;      // MeV
      Timing timing;
      double lifetime;    // ns
      double meanOn;      // photons per on-spill microspill
      double meanOff;     // photons per off-spill microspill
      double meanFlat;    // photons per microspill, time-averaged
      TH1F*  hTime;
    };

    std::vector<Line>  lines_;
    CLHEP::Hep3Vector  centre_;
    CLHEP::Hep3Vector  aim_, u_, v_;     // aim and two perpendicular unit vectors
    double             beamRadius_;
    double             cosDivergence_;
    double             stopTimeMean_, stopTimeSigma_;
    std::vector<double> stopCdf_, stopLow_, stopWidth_;   // empty = Gaussian
    SpillMode          spillMode_;
    double             supercycle_, spillOn_, micropulse_;
    unsigned long      firstMicrospill_;

    art::RandomNumberGenerator::base_engine_t& eng_;
    CLHEP::RandFlat        randFlat_;
    CLHEP::RandGaussQ      randGauss_;
    CLHEP::RandExponential randExp_;
    CLHEP::RandPoissonQ    randPoisson_;

    TH1F* hLine_          = nullptr;
    TH1F* hNPerMicrospill_ = nullptr;
    TH1F* hSpillPhase_    = nullptr;
    TH2F* hBeamSpot_      = nullptr;

    void   loadStopTimeHist(const std::string& file, const std::string& name);
    double stopTime();
    CLHEP::Hep3Vector direction();
  };

  // ---------------------------------------------------------------------------

  STMLineSourceGun::STMLineSourceGun(const Parameters& conf)
    : art::EDProducer(conf)
    , centre_(conf().x(), conf().y(), conf().z())
    , beamRadius_(conf().beamRadius())
    , stopTimeMean_(conf().stopTimeMean())
    , stopTimeSigma_(conf().stopTimeSigma())
    , supercycle_(conf().supercycle())
    , spillOn_(conf().spillOn())
    , micropulse_(conf().micropulse())
    , firstMicrospill_(conf().firstMicrospill())
    , eng_(createEngine(art::ServiceHandle<SeedService>()->getSeed()))
    , randFlat_(eng_)
    , randGauss_(eng_)
    , randExp_(eng_)
    , randPoisson_(eng_)
  {
    produces<GenParticleCollection>();

    // Beam geometry
    const double amag = std::sqrt(conf().aimX()*conf().aimX() + conf().aimY()*conf().aimY() +
                                  conf().aimZ()*conf().aimZ());
    if (amag < std::numeric_limits<double>::epsilon())
      throw cet::exception("BADCONFIG") << "STMLineSourceGun: aim direction has zero magnitude\n";
    aim_.set(conf().aimX()/amag, conf().aimY()/amag, conf().aimZ()/amag);
    u_ = (std::abs(aim_.x()) < 0.9) ? aim_.cross(CLHEP::Hep3Vector(1, 0, 0)).unit()
                                    : aim_.cross(CLHEP::Hep3Vector(0, 1, 0)).unit();
    v_ = aim_.cross(u_);
    if (beamRadius_ < 0.0)
      throw cet::exception("BADCONFIG") << "STMLineSourceGun: beamRadius must be >= 0\n";
    if (conf().divergence() < 0.0 || conf().divergence() > 180.0)
      throw cet::exception("BADCONFIG") << "STMLineSourceGun: divergence must be in [0,180] deg\n";
    cosDivergence_ = std::cos(conf().divergence() * CLHEP::degree);

    // Spill structure
    const std::string mode = conf().spillMode();
    if      (mode == "onSpill")    spillMode_ = SpillMode::onSpill;
    else if (mode == "supercycle") spillMode_ = SpillMode::supercycle;
    else if (mode == "flat")       spillMode_ = SpillMode::flat;
    else throw cet::exception("BADCONFIG")
           << "STMLineSourceGun: spillMode must be \"onSpill\", \"supercycle\" or \"flat\", got \"" << mode << "\"\n";
    if (micropulse_ <= 0.0 || supercycle_ <= 0.0 || spillOn_ <= 0.0 || spillOn_ > supercycle_)
      throw cet::exception("BADCONFIG")
        << "STMLineSourceGun: need micropulse > 0 and 0 < spillOn <= supercycle\n";
    const double duty = spillOn_ / supercycle_;

    // Stop-time model
    std::string histFile, histName;
    const bool hasFile = conf().stopTimeHistFile(histFile);
    const bool hasName = conf().stopTimeHistName(histName);
    if (hasFile != hasName)
      throw cet::exception("BADCONFIG")
        << "STMLineSourceGun: stopTimeHistFile and stopTimeHistName must be given together\n";
    if (hasFile) loadStopTimeHist(histFile, histName);
    else if (stopTimeSigma_ < 0.0)
      throw cet::exception("BADCONFIG") << "STMLineSourceGun: stopTimeSigma must be >= 0\n";

    // Lines and per-line Poisson means
    const double total = conf().meanPhotonsPerMicrospill();
    if (total <= 0.0)
      throw cet::exception("BADCONFIG") << "STMLineSourceGun: meanPhotonsPerMicrospill must be > 0\n";
    double wsum = 0.0;
    for (const auto& lc : conf().lines()) {
      if (lc.weight() < 0.0)
        throw cet::exception("BADCONFIG") << "STMLineSourceGun: line weights must be >= 0\n";
      if (lc.energy() <= 0.0)
        throw cet::exception("BADCONFIG") << "STMLineSourceGun: line energies must be > 0\n";
      wsum += lc.weight();
    }
    if (wsum <= 0.0)
      throw cet::exception("BADCONFIG") << "STMLineSourceGun: need at least one line with weight > 0\n";

    for (const auto& lc : conf().lines()) {
      Line l{};
      l.energy = lc.energy();
      const std::string t = lc.timing();
      if      (t == "prompt")     l.timing = Timing::prompt;
      else if (t == "delayed")    l.timing = Timing::delayed;
      else if (t == "continuous") l.timing = Timing::continuous;
      else throw cet::exception("BADCONFIG")
             << "STMLineSourceGun: timing must be \"prompt\", \"delayed\" or \"continuous\", got \"" << t << "\"\n";
      l.lifetime = 0.0;
      if (l.timing == Timing::delayed) {
        if (!lc.lifetime(l.lifetime) || l.lifetime <= 0.0)
          throw cet::exception("BADCONFIG")
            << "STMLineSourceGun: line at " << l.energy << " MeV is \"delayed\" and needs lifetime > 0 [ns]\n";
      }
      l.meanFlat = total * lc.weight() / wsum;
      const bool beamCorrelated = (l.timing != Timing::continuous);
      l.meanOn  = beamCorrelated ? l.meanFlat / duty : l.meanFlat;
      l.meanOff = beamCorrelated ? 0.0 : l.meanFlat;
      l.hTime   = nullptr;
      lines_.push_back(l);
    }

    if (conf().doHistograms()) {
      // Booked in the module-label directory, e.g. generate/hLine
      art::ServiceHandle<art::TFileService> tfs;
      hLine_ = tfs->make<TH1F>("hLine", "Generated line;line index;photons",
                              static_cast<int>(lines_.size()), -0.5, lines_.size() - 0.5);
      hNPerMicrospill_ = tfs->make<TH1F>("hNPerMicrospill", "Photons per microspill;N;microspills", 20, -0.5, 19.5);
      hSpillPhase_ = tfs->make<TH1F>("hSpillPhase", "Microspill phase in supercycle;phase [ms];microspills",
                                    200, 0., supercycle_ * 1e-6);
      hBeamSpot_ = tfs->make<TH2F>("hBeamSpot", "Start position in the beam plane;u [mm];v [mm]",
                                  100, -1.1*beamRadius_ - 1e-3, 1.1*beamRadius_ + 1e-3,
                                  100, -1.1*beamRadius_ - 1e-3, 1.1*beamRadius_ + 1e-3);
      for (size_t i = 0; i < lines_.size(); ++i) {
        const double keV = lines_[i].energy * 1e3;
        const TString name  = TString::Format("hTime_%.0fkeV", keV);
        const TString title = TString::Format("%.1f keV emission time;t in microspill [ns];photons", keV);
        lines_[i].hTime = tfs->make<TH1F>(name.Data(), title.Data(), 170, 0., micropulse_);
      }
    }

    // Summary: on-spill instantaneous rates are what drive pile-up
    const double kHzPerMean = 1e6 / micropulse_;   // photons/microspill -> kHz
    std::ostringstream os;
    os << "beam centre (" << centre_.x() << ", " << centre_.y() << ", " << centre_.z() << ") mm"
       << "  aim (" << aim_.x() << ", " << aim_.y() << ", " << aim_.z() << ")"
       << "  radius " << beamRadius_ << " mm  divergence " << conf().divergence() << " deg\n"
       << "  spillMode " << mode << "  duty " << duty
       << "  stop time " << (stopCdf_.empty() ? "Gaussian" : "histogram " + histName)
       << (stopCdf_.empty() ? " (" + std::to_string(stopTimeMean_) + " +- " + std::to_string(stopTimeSigma_) + " ns)" : std::string())
       << "\n  line [keV]  timing       avg [kHz]  on-spill [kHz]  off-spill [kHz]\n";
    for (const auto& l : lines_) {
      const char* tn = (l.timing == Timing::prompt) ? "prompt" : (l.timing == Timing::delayed) ? "delayed" : "continuous";
      os << "  " << l.energy * 1e3 << "  " << tn
         << "  " << l.meanFlat * kHzPerMean << "  " << l.meanOn * kHzPerMean << "  " << l.meanOff * kHzPerMean << "\n";
    }
    mf::LogInfo("STMLineSourceGun") << os.str();
  }

  // ---------------------------------------------------------------------------

  void STMLineSourceGun::loadStopTimeHist(const std::string& file, const std::string& name) {
    std::unique_ptr<TFile> f(TFile::Open(file.c_str(), "READ"));
    if (!f || f->IsZombie())
      throw cet::exception("BADCONFIG") << "STMLineSourceGun: cannot open stopTimeHistFile " << file << "\n";
    TH1* h = dynamic_cast<TH1*>(f->Get(name.c_str()));
    if (!h)
      throw cet::exception("BADCONFIG") << "STMLineSourceGun: no TH1 \"" << name << "\" in " << file << "\n";
    double running = 0.0;
    for (int b = 1; b <= h->GetNbinsX(); ++b) {
      const double c = h->GetBinContent(b);
      if (c <= 0.0) continue;
      running += c;
      stopCdf_.push_back(running);
      stopLow_.push_back(h->GetXaxis()->GetBinLowEdge(b));
      stopWidth_.push_back(h->GetXaxis()->GetBinWidth(b));
    }
    if (running <= 0.0)
      throw cet::exception("BADCONFIG") << "STMLineSourceGun: stop-time histogram " << name << " is empty\n";
    for (auto& c : stopCdf_) c /= running;
    stopCdf_.back() = 1.0;
  }

  double STMLineSourceGun::stopTime() {
    if (stopCdf_.empty()) return randGauss_.fire(stopTimeMean_, stopTimeSigma_);
    const double r = randFlat_.fire();
    const size_t b = std::min<size_t>(std::upper_bound(stopCdf_.begin(), stopCdf_.end(), r) - stopCdf_.begin(),
                                      stopCdf_.size() - 1);
    return stopLow_[b] + stopWidth_[b] * randFlat_.fire();
  }

  CLHEP::Hep3Vector STMLineSourceGun::direction() {
    if (cosDivergence_ >= 1.0) return aim_;
    const double cosTheta = cosDivergence_ + (1.0 - cosDivergence_) * randFlat_.fire();
    const double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta*cosTheta));
    const double phi      = CLHEP::twopi * randFlat_.fire();
    return sinTheta*std::cos(phi)*u_ + sinTheta*std::sin(phi)*v_ + cosTheta*aim_;
  }

  // ---------------------------------------------------------------------------

  void STMLineSourceGun::produce(art::Event& event) {
    bool onSpill = true;
    if (spillMode_ == SpillMode::supercycle) {
      const unsigned long iMicrospill = firstMicrospill_ + event.id().event() - 1;
      const double phase = std::fmod(static_cast<double>(iMicrospill) * micropulse_, supercycle_);
      onSpill = phase < spillOn_;
      if (hSpillPhase_) hSpillPhase_->Fill(phase * 1e-6);
    }

    auto output = std::make_unique<GenParticleCollection>();

    for (size_t i = 0; i < lines_.size(); ++i) {
      const Line& l = lines_[i];
      const double mean = (spillMode_ == SpillMode::flat) ? l.meanFlat : (onSpill ? l.meanOn : l.meanOff);
      if (mean <= 0.0) continue;
      const long n = randPoisson_.fire(mean);

      for (long k = 0; k < n; ++k) {
        double t = 0.0;
        switch (l.timing) {
          case Timing::prompt:     t = stopTime();                              break;
          case Timing::delayed:    t = stopTime() + randExp_.fire(l.lifetime);  break;
          case Timing::continuous: t = micropulse_ * randFlat_.fire();          break;
        }
        t = std::fmod(t, micropulse_);
        if (t < 0.0) t += micropulse_;

        // Uniform disk perpendicular to the beam axis
        const double r   = beamRadius_ * std::sqrt(randFlat_.fire());
        const double phi = CLHEP::twopi * randFlat_.fire();
        const double du  = r * std::cos(phi), dv = r * std::sin(phi);
        const CLHEP::Hep3Vector pos = centre_ + du*u_ + dv*v_;

        const CLHEP::HepLorentzVector mom(l.energy * direction(), l.energy);
        output->emplace_back(PDGCode::gamma, GenId::particleGun, pos, mom, t);

        if (hLine_)     hLine_->Fill(i);
        if (l.hTime)    l.hTime->Fill(t);
        if (hBeamSpot_) hBeamSpot_->Fill(du, dv);
      }
    }

    if (hNPerMicrospill_) hNPerMicrospill_->Fill(output->size());
    event.put(std::move(output));
  }

} // namespace mu2e

DEFINE_ART_MODULE(mu2e::STMLineSourceGun)
