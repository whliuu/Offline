// Records per-generated-photon kinematics and energy deposited in the HPGe
// crystal, for mapping geometric acceptance vs. incidence angle and position.
//
// TTree branches:
//   eventId                -- art event number (for joining with downstream
//                             digitization/MWD trees; assumes 1 photon/event)
//   genX, genY, genZ      -- generator position (mm)
//   genPx, genPy, genPz   -- unit momentum direction
//   genE                   -- generated kinetic energy (MeV)
//   crystalEdep            -- total ionising energy deposit in crystal (MeV)
//   hitCrystal             -- bool: any step landed in crystal
//   localX, localY        -- crystal-local (x, y) of first step in crystal (mm)
//   localR, localZ         -- crystal-local impact point of first step in crystal (mm)
//   cosTheta               -- cos(angle) of photon wrt crystal axis
//
// Original author: Leo Liu

#include <cmath>

#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"

#include "cetlib_except/exception.h"
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/types/Atom.h"

#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"
#include "Offline/STMGeom/inc/HPGeDetector.hh"
#include "Offline/STMGeom/inc/STM.hh"

#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include "art_root_io/TFileService.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"

namespace mu2e {

  class HPGeAcceptance : public art::EDAnalyzer {
  public:
    using Name = fhicl::Name;
    using Comment = fhicl::Comment;
    struct Config {
      fhicl::Atom<art::InputTag> genParticleTag{Name("GenParticleTag"), Comment("Tag for GenParticleCollection"), "generate"};
      fhicl::Atom<art::InputTag> stepPointMCsTag{Name("StepPointMCsTag"), Comment("Tag for StepPointMCs in STMDet"), "g4run:STMDet"};
      fhicl::Atom<bool> writeSteps{Name("WriteSteps"), Comment("Also write a per-step TTree (crystal-local x,y,z,edep,pdg) for 3D track display"), false};
    };
    using Parameters = art::EDAnalyzer::Table<Config>;
    explicit HPGeAcceptance(const Parameters& conf);
    void beginRun(const art::Run& run) override;
    void analyze(const art::Event& event) override;

  private:
    bool stepInCrystal(const CLHEP::Hep3Vector& worldPos,
                       double& localX_out, double& localY_out,
                       double& localR_out, double& localZ_out) const;

    art::ProductToken<GenParticleCollection> genToken_;
    art::ProductToken<StepPointMCCollection> stepToken_;
    bool writeSteps_;

    // Crystal geometry
    CLHEP::Hep3Vector crystalOrigin_{-3986.30, 0.0, 40612.70};
    CLHEP::HepRotation crystalRotation_ = CLHEP::HepRotation::IDENTITY;
    CLHEP::Hep3Vector crystalAxis_{-std::sin(45.0 * CLHEP::deg), 0.0, std::cos(45.0 * CLHEP::deg)};
    double crystalR_ = 36.05;
    double crystalL_ = 78.5;
    double crystalTol_ = 0.1;
    bool geomLoaded_ = false;
    double xBeamCentre_ = -3904.0;

    // TTree
    TTree* tree_ = nullptr;
    unsigned int eventId_;
    double genX_, genY_, genZ_;
    double genPx_, genPy_, genPz_;
    double genE_;
    double crystalEdep_;
    int hitCrystal_;
    double localX_, localY_;
    double localR_, localZ_;
    double cosTheta_;

    // Per-step TTree (one row per in-crystal G4 step), for 3D track display.
    TTree* stepTree_ = nullptr;
    unsigned int sEventId_;
    double sX_, sY_, sZ_;        // crystal-local position (mm), Z=0 at front face
    double sEdep_;               // ionising energy deposit at this step (MeV)
    double sTime_;               // global step time (ns)
    int    sPdg_;                // PDG id of the depositing track
    int    sTrackId_;            // SimParticle id of the depositing track
    int    sParentId_;           // SimParticle id of its parent (0 if primary)
    int    sIsPrimary_;          // 1 if this track is the generated photon (no parent)

    // Summary histograms
    TH1D* hCosTheta_all_ = nullptr;
    TH1D* hCosTheta_hit_ = nullptr;
    TH1D* hCosTheta_photopeak_ = nullptr;
    TH2D* hLocalRZ_hit_ = nullptr;

    unsigned long nGenerated_ = 0;
    unsigned long nHitCrystal_ = 0;
  };

  HPGeAcceptance::HPGeAcceptance(const Parameters& conf)
    : art::EDAnalyzer(conf),
      genToken_(consumes<GenParticleCollection>(conf().genParticleTag())),
      stepToken_(consumes<StepPointMCCollection>(conf().stepPointMCsTag())),
      writeSteps_(conf().writeSteps())
  {
    art::ServiceHandle<art::TFileService> tfs;
    tree_ = tfs->make<TTree>("acceptance", "HPGe geometric acceptance");
    tree_->Branch("eventId", &eventId_, "eventId/i");
    tree_->Branch("genX", &genX_, "genX/D");
    tree_->Branch("genY", &genY_, "genY/D");
    tree_->Branch("genZ", &genZ_, "genZ/D");
    tree_->Branch("genPx", &genPx_, "genPx/D");
    tree_->Branch("genPy", &genPy_, "genPy/D");
    tree_->Branch("genPz", &genPz_, "genPz/D");
    tree_->Branch("genE", &genE_, "genE/D");
    tree_->Branch("crystalEdep", &crystalEdep_, "crystalEdep/D");
    tree_->Branch("hitCrystal", &hitCrystal_, "hitCrystal/I");
    tree_->Branch("localX", &localX_, "localX/D");
    tree_->Branch("localY", &localY_, "localY/D");
    tree_->Branch("localR", &localR_, "localR/D");
    tree_->Branch("localZ", &localZ_, "localZ/D");
    tree_->Branch("cosTheta", &cosTheta_, "cosTheta/D");

    if (writeSteps_) {
      stepTree_ = tfs->make<TTree>("steps", "HPGe in-crystal steps (3D track display)");
      stepTree_->Branch("eventId",   &sEventId_,  "eventId/i");
      stepTree_->Branch("x",         &sX_,        "x/D");
      stepTree_->Branch("y",         &sY_,        "y/D");
      stepTree_->Branch("z",         &sZ_,        "z/D");
      stepTree_->Branch("edep",      &sEdep_,     "edep/D");
      stepTree_->Branch("time",      &sTime_,     "time/D");
      stepTree_->Branch("pdg",       &sPdg_,      "pdg/I");
      stepTree_->Branch("trackId",   &sTrackId_,  "trackId/I");
      stepTree_->Branch("parentId",  &sParentId_, "parentId/I");
      stepTree_->Branch("isPrimary", &sIsPrimary_,"isPrimary/I");
    }

    hCosTheta_all_ = tfs->make<TH1D>("hCosTheta_all",
        "cos(#theta) wrt crystal axis, all generated;cos #theta;counts",
        100, 0.0, 1.0);
    hCosTheta_hit_ = tfs->make<TH1D>("hCosTheta_hit",
        "cos(#theta) wrt crystal axis, hit crystal;cos #theta;counts",
        100, 0.0, 1.0);
    hCosTheta_photopeak_ = tfs->make<TH1D>("hCosTheta_photopeak",
        "cos(#theta) wrt crystal axis, photopeak;cos #theta;counts",
        100, 0.0, 1.0);
    hLocalRZ_hit_ = tfs->make<TH2D>("hLocalRZ_hit",
        "Crystal-local (R, Z) of first hit;R [mm];Z [mm]",
        50, 0, 40, 80, 0, 80);
  }

  void HPGeAcceptance::beginRun(const art::Run&) {
    try {
      GeomHandle<STM> stm;
      HPGeDetector const* hpge = stm->getHPGeDetectorPtr();
      if (hpge) {
        crystalRotation_ = hpge->rotation();
        crystalR_ = hpge->CrystalR();
        crystalL_ = hpge->CrystalL();
        geomLoaded_ = true;
        // Crystal axis = rotation applied to local +z
        crystalAxis_ = crystalRotation_ * CLHEP::Hep3Vector(0, 0, 1);
      }
    } catch (std::exception const& e) {
      std::cout << "HPGeAcceptance: geometry query failed, using hardcoded values" << std::endl;
    }
  }

  bool HPGeAcceptance::stepInCrystal(const CLHEP::Hep3Vector& worldPos,
                                      double& localX_out, double& localY_out,
                                      double& localR_out, double& localZ_out) const {
    CLHEP::Hep3Vector local = worldPos - crystalOrigin_;
    if (geomLoaded_) {
      local = crystalRotation_.inverse() * local;
    } else {
      local.rotateY(45.0 * CLHEP::degree);
    }
    // Shift so Z=0 is the front face
    local.setZ(local.z() + (crystalL_ / 2.0));
    localX_out = local.x();
    localY_out = local.y();
    localR_out = local.perp();
    localZ_out = local.z();
    return (localZ_out >= -crystalTol_ && localZ_out <= crystalL_ + crystalTol_
            && localR_out <= crystalR_ + crystalTol_);
  }

  void HPGeAcceptance::analyze(const art::Event& event) {
    auto const& genParticles = event.getProduct(genToken_);
    auto const& steps = event.getProduct(stepToken_);

    for (auto const& gen : genParticles) {
      nGenerated_++;

      eventId_ = event.id().event();
      genX_ = gen.position().x();
      genY_ = gen.position().y();
      genZ_ = gen.position().z();

      CLHEP::Hep3Vector pDir = gen.momentum().vect().unit();
      genPx_ = pDir.x();
      genPy_ = pDir.y();
      genPz_ = pDir.z();
      genE_ = gen.momentum().e() - gen.momentum().m();

      // cos(angle) between photon direction and crystal axis
      cosTheta_ = pDir.dot(crystalAxis_);

      // Sum energy deposited by this photon's shower in the crystal
      crystalEdep_ = 0.0;
      hitCrystal_ = 0;
      localX_ = 0.0;
      localY_ = 0.0;
      localR_ = -1.0;
      localZ_ = -1.0;
      bool firstHit = true;

      for (auto const& step : steps) {
        if (step.position().x() > xBeamCentre_) continue;
        double tmpX, tmpY, tmpR, tmpZ;
        if (stepInCrystal(step.position(), tmpX, tmpY, tmpR, tmpZ)) {
          crystalEdep_ += step.ionizingEdep();
          hitCrystal_ = 1;
          if (firstHit) {
            localX_ = tmpX;
            localY_ = tmpY;
            localR_ = tmpR;
            localZ_ = tmpZ;
            firstHit = false;
          }
          if (writeSteps_) {
            sEventId_   = eventId_;
            sX_         = tmpX;
            sY_         = tmpY;
            sZ_         = tmpZ;
            sEdep_      = step.ionizingEdep();
            sTime_      = step.time();
            auto const& sp = step.simParticle();
            sPdg_       = sp ? (int)sp->pdgId() : 0;
            sTrackId_   = sp ? (int)sp->id().asUint() : -1;
            sParentId_  = sp ? (int)sp->parentId().asUint() : 0;
            sIsPrimary_ = (sp && sp->parent().isNull()) ? 1 : 0;
            stepTree_->Fill();
          }
        }
      }

      hCosTheta_all_->Fill(cosTheta_);
      if (hitCrystal_) {
        nHitCrystal_++;
        hCosTheta_hit_->Fill(cosTheta_);
        hLocalRZ_hit_->Fill(localR_, localZ_);
        // Photopeak: deposited > 95% of generated energy
        if (crystalEdep_ > 0.95 * genE_) {
          hCosTheta_photopeak_->Fill(cosTheta_);
        }
      }

      tree_->Fill();
    }
  }

} // namespace mu2e

DEFINE_ART_MODULE(mu2e::HPGeAcceptance)
