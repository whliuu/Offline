// Records per-generated-photon kinematics and energy deposited in the HPGe
// crystal, for mapping geometric acceptance vs. incidence angle and position.
//
// TTree branches:
//   genX, genY, genZ      -- generator position (mm)
//   genPx, genPy, genPz   -- unit momentum direction
//   genE                   -- generated kinetic energy (MeV)
//   crystalEdep            -- total ionising energy deposit in crystal (MeV)
//   hitCrystal             -- bool: any step landed in crystal
//   localR, localZ         -- crystal-local impact point of first step in crystal (mm)
//   cosTheta               -- cos(angle) of photon wrt crystal axis
//
// Original author: Weihan Liu

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
    };
    using Parameters = art::EDAnalyzer::Table<Config>;
    explicit HPGeAcceptance(const Parameters& conf);
    void beginRun(const art::Run& run) override;
    void analyze(const art::Event& event) override;

  private:
    bool stepInCrystal(const CLHEP::Hep3Vector& worldPos,
                       double& localR_out, double& localZ_out) const;

    art::ProductToken<GenParticleCollection> genToken_;
    art::ProductToken<StepPointMCCollection> stepToken_;

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
    double genX_, genY_, genZ_;
    double genPx_, genPy_, genPz_;
    double genE_;
    double crystalEdep_;
    int hitCrystal_;
    double localR_, localZ_;
    double cosTheta_;

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
      stepToken_(consumes<StepPointMCCollection>(conf().stepPointMCsTag()))
  {
    art::ServiceHandle<art::TFileService> tfs;
    tree_ = tfs->make<TTree>("acceptance", "HPGe geometric acceptance");
    tree_->Branch("genX", &genX_, "genX/D");
    tree_->Branch("genY", &genY_, "genY/D");
    tree_->Branch("genZ", &genZ_, "genZ/D");
    tree_->Branch("genPx", &genPx_, "genPx/D");
    tree_->Branch("genPy", &genPy_, "genPy/D");
    tree_->Branch("genPz", &genPz_, "genPz/D");
    tree_->Branch("genE", &genE_, "genE/D");
    tree_->Branch("crystalEdep", &crystalEdep_, "crystalEdep/D");
    tree_->Branch("hitCrystal", &hitCrystal_, "hitCrystal/I");
    tree_->Branch("localR", &localR_, "localR/D");
    tree_->Branch("localZ", &localZ_, "localZ/D");
    tree_->Branch("cosTheta", &cosTheta_, "cosTheta/D");

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
                                      double& localR_out, double& localZ_out) const {
    CLHEP::Hep3Vector local = worldPos - crystalOrigin_;
    if (geomLoaded_) {
      local = crystalRotation_.inverse() * local;
    } else {
      local.rotateY(45.0 * CLHEP::degree);
    }
    // Shift so Z=0 is the front face
    local.setZ(local.z() + (crystalL_ / 2.0));
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
      localR_ = -1.0;
      localZ_ = -1.0;
      bool firstHit = true;

      for (auto const& step : steps) {
        if (step.position().x() > xBeamCentre_) continue;
        double tmpR, tmpZ;
        if (stepInCrystal(step.position(), tmpR, tmpZ)) {
          crystalEdep_ += step.ionizingEdep();
          hitCrystal_ = 1;
          if (firstHit) {
            localR_ = tmpR;
            localZ_ = tmpZ;
            firstHit = false;
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
