// Adapted from ReadVirtualDetector_module.cc
// For StepPointMCs in STMDet, generates a TTree with energy in branch "E" and time in branch "time". For a chosen STM detector
//  - Iterate over the StepPointMCs, determine the associated SimParticle
//  - Find the most parent particle of the associated SimParticle within the associated STMDet
//  - Increment the energy associated with the parent particle
//    - If the parent particle ID doesn't exist in the collection, add a new entry to the parent particle ID vector, energy deposited map, and time map
//  - Determine the PDG ID of the top particles, increment the mapped counter
//    - If the PDG ID entry does not exist in the map, create a new entry.
//  - Print the PDG ID counts at the end of the job
// Input Parameters
//  - Detector - either "HPGe" or "LaBr" - applies a position cut to calculate the energy deposited by a particle going through the chosen detector
//  - StepPointMCsTag - tag of data product containing the StepPoints for STMDet
//  - SimParticlemvTag - tag of data product containing the SimParticles for STMDet
// Original author: Ivan Logashenko
// Adapted by: Pawel Plesniak & Leo Liu

// stdlib includes
#include <limits>

// art includes
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Principal/Run.h"

// exception handling
#include "cetlib_except/exception.h"

// fhicl includes
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/ParameterSet.h"

// message handling
#include "messagefacility/MessageLogger/MessageLogger.h"

// Offline includes
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "Offline/GlobalConstantsService/inc/ParticleDataList.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"
#include "Offline/STMGeom/inc/HPGeDetector.hh"
#include "Offline/STMGeom/inc/STM.hh"

// CLHEP includes
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Units/PhysicalConstants.h"

// ROOT includes
#include "art_root_io/TFileService.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TTree.h"


// Mu2e type definitions
typedef cet::map_vector_key key_type;
typedef unsigned long VolumeId_type;

namespace mu2e {
  class HPGeTree : public art::EDAnalyzer {
    public:
      using Name=fhicl::Name;
      using Comment=fhicl::Comment;
      struct Config {
        fhicl::Atom<std::string> detector{ Name("Detector"), Comment("Which detector to generate energy histograms for, either 'HPGe' or 'LaBr'")};
        fhicl::Atom<art::InputTag> stepPointMCsTag{ Name("StepPointMCsTag"), Comment("Tag identifying the StepPointMCs")};
        fhicl::Atom<art::InputTag> simParticlemvTag{ Name("SimParticlemvTag"), Comment("Tag identifying the SimParticlemv")};
      };
      using Parameters = art::EDAnalyzer::Table<Config>;
      explicit HPGeTree(const Parameters& conf);
      std::tuple<key_type, int> topParent(std::set<key_type>& SimParticleIDs, const SimParticle particle);
      double parentTime(const art::Event& event, key_type parentId);
      void beginRun(const art::Run& run) override;
      void analyze(const art::Event& event);
      void endJob();
      bool stepInCrystal(const CLHEP::Hep3Vector& worldPos) const;
    private:
      std::string detector = "";
      std::vector<std::string> detectors{"HPGe", "LaBr"};

      SimParticle stepParticle;
      std::set<key_type> SimParticleIds;

      key_type topParentId;
      std::vector<key_type> topParentIds;
      std::vector<key_type>::iterator topParentIdsIt;

      std::map<int, int> pdgIds; // <ID, count>
      std::map<int, int>::iterator pdgIdsIt;

      // EDeps = <ID, deposited energy>, times = <ID, time>
      // "AllHalf" = any hit on the HPGe side of STMDet (x < xBeamCentre)
      // "Crystal" = additionally required to lie inside the crystal cylinder
      std::map<key_type, double> EDepsAllHalf, EDepsCrystal, times;
      std::map<key_type, bool>   inCrystalFlag;   // did any step of this parent land inside crystal?

      TTree* ttree = nullptr;          // all HPGe-half hits (legacy behaviour)
      TTree* ttreeCrystal = nullptr;   // parents with at least one step inside the crystal cylinder
      double xBeamCentre = -3904.0;
      int pdgId = 0;
      double E = 0.0, time = 0.0;
      double ECrystal = 0.0;            // per-parent crystal-only energy sum, written to ttreeCrystal

      // Diagnostic histograms of step world (x,z), filled per-step
      TH2D* hStepXZ_half = nullptr;     // steps in HPGe half
      TH2D* hStepXZ_crystal = nullptr;  // subset that fall inside the crystal cylinder
      TH1D* hLocalR = nullptr;          // crystal-local R for HPGe-half steps
      TH1D* hLocalZ = nullptr;          // crystal-local Z for HPGe-half steps

      // Crystal geometry, populated from GeomHandle<STM> at beginRun.
      // Fallback to values verified against constructSTM.cc printout (2026-04-20).
      CLHEP::Hep3Vector crystalOrigin{-3986.30, 0.0, 40612.70};
      CLHEP::HepRotation crystalRotation = CLHEP::HepRotation::IDENTITY; // world rotation of crystal
      double crystalR = 36.05;
      double crystalL = 78.5;
      double crystalTol = 0.1;
      bool crystalGeomLoaded = false;

      art::ProductToken<StepPointMCCollection> StepPointMCsToken;
      art::ProductToken<SimParticleCollection> SimParticlemvToken;
      art::Ptr<SimParticle> parent;
  };

  HPGeTree::HPGeTree(const Parameters& conf) :
    art::EDAnalyzer(conf),
    detector(conf().detector()),
    StepPointMCsToken(consumes<StepPointMCCollection>(conf().stepPointMCsTag())),
    SimParticlemvToken(consumes<SimParticleCollection>(conf().simParticlemvTag())) {
      // Check if detector is one of the allowed types
      if (std::find(detectors.begin(), detectors.end(), detector) == detectors.end())
        throw cet::exception("Configuration") << "'detector' must be one of 'HPGe' or 'LaBr'";
      // TODO - remove this when the LaBr shower energy is defined
      if (detector == "LaBr")
        throw cet::exception("Configuration") << "Currently this code only works for HPGe, exiting.\n";

      // Set up TTree
      art::ServiceHandle<art::TFileService> tfs;
      ttree = tfs->make<TTree>( "ttree", "Detector ttree");
      ttree->Branch("E", &E, "E/D");
      ttree->Branch("time", &time, "time/D");

      // Crystal-only TTree: per-parent energy sum, restricted to steps inside the
      // HPGe crystal cylinder (world pos, translated + rotated into crystal local).
      ttreeCrystal = tfs->make<TTree>("ttreeCrystal", "Crystal-only detector ttree");
      ttreeCrystal->Branch("E", &ECrystal, "E/D");
      ttreeCrystal->Branch("time", &time, "time/D");

      // Diagnostic histograms of step world (x,z) for all HPGe-half steps vs
      // the subset that lie inside the crystal cylinder.
      art::TFileDirectory diag = tfs->mkdir("HPGeTreeDiag");
      hStepXZ_half    = diag.make<TH2D>("hStepXZ_half",
          "Step world (x,z) on HPGe half of STMDet;x [mm];z [mm]",
          200, -4050, -3850, 400, 40550, 40850);
      hStepXZ_crystal = diag.make<TH2D>("hStepXZ_crystal",
          "Step world (x,z) inside crystal cylinder;x [mm];z [mm]",
          200, -4050, -3850, 400, 40550, 40850);
      hLocalR = diag.make<TH1D>("hLocalR",
          "Crystal-local R of HPGe-half steps;R [mm];steps", 200, 0, 200);
      hLocalZ = diag.make<TH1D>("hLocalZ",
          "Crystal-local Z of HPGe-half steps;Z [mm];steps", 400, -200, 200);
  };

  void HPGeTree::beginRun(const art::Run&) {
    try {
      GeomHandle<STM> stm;
      HPGeDetector const* hpge = stm->getHPGeDetectorPtr();
      if (hpge) {
        // HPGeDetector::originInMu2e() returns the envelope (endcap) origin,
        // NOT the crystal centre. Ground truth from constructSTM.cc (2026-04-20)
        // gives crystal centre = (-3986.30, 0, 40612.70). Keep the fallback
        // hardcoded value rather than using the envelope origin.
        crystalRotation = hpge->rotation();
        crystalR        = hpge->CrystalR();
        crystalL        = hpge->CrystalL();
        crystalGeomLoaded = true;
        std::cout << "===== HPGeTree geometry =====" << std::endl;
        std::cout << "  Crystal origin (mm): (" << crystalOrigin.x() << ", "
                  << crystalOrigin.y() << ", " << crystalOrigin.z() << ")" << std::endl;
        std::cout << "  Crystal R, L (mm):   " << crystalR << ", " << crystalL << std::endl;
        std::cout << "  Rotation matrix:" << std::endl;
        std::cout << "    [" << crystalRotation.xx() << ", " << crystalRotation.xy() << ", " << crystalRotation.xz() << "]" << std::endl;
        std::cout << "    [" << crystalRotation.yx() << ", " << crystalRotation.yy() << ", " << crystalRotation.yz() << "]" << std::endl;
        std::cout << "    [" << crystalRotation.zx() << ", " << crystalRotation.zy() << ", " << crystalRotation.zz() << "]" << std::endl;
        std::cout << "=============================" << std::endl;
      }
    } catch (std::exception const& e) {
      std::cout << "HPGeTree beginRun: geometry query failed: " << e.what()
                << " -- falling back to hardcoded crystal constants" << std::endl;
    }
  };

  bool HPGeTree::stepInCrystal(const CLHEP::Hep3Vector& worldPos) const {
    // Transform the world position into crystal-local coordinates and check
    // the cylinder envelope. We apply the inverse of the crystal rotation; if
    // the rotation is unavailable we fall back to a fixed rotateY(+45) that
    // matches the hardcoded values in HPGeWaveformsFromStepPointMCs.
    CLHEP::Hep3Vector local = worldPos - crystalOrigin;
    if (crystalGeomLoaded) {
      local = crystalRotation.inverse() * local;
    } else {
      local.rotateY(45.0 * CLHEP::degree);
    }
    local.setZ(local.z() + (crystalL / 2.0));
    double R = local.perp();
    double Z = local.z();
    if (hLocalR) hLocalR->Fill(R);
    if (hLocalZ) hLocalZ->Fill(Z);
    double maxR = crystalR + crystalTol;
    double maxZ = crystalL + crystalTol;
    return (Z >= -crystalTol && Z <= maxZ && R <= maxR);
  };

  std::tuple<key_type, int> HPGeTree::topParent(std::set<key_type>& SimParticleIds, const SimParticle particle) {
    // Get the particle parent
    parent = particle.parent();

    // If the passed particle is a primary (no parent), return its own ID and PDG ID
    if (!parent.isNonnull())
      return std::make_tuple(particle.id(), particle.pdgId());

    // If the passed particle has no parent in STMDet, return its ID and PDG ID
    if (std::find(SimParticleIds.begin(), SimParticleIds.end(), parent->id()) == SimParticleIds.end())
      return std::make_tuple(particle.id(), particle.pdgId());

    // If the particle has a parent in STMDet, update the particle parent
    while (parent->parent().isNonnull() &&
           std::find(SimParticleIds.begin(), SimParticleIds.end(), parent->parent()->id()) != SimParticleIds.end())
      parent = parent->parent();

    // Return the ID and PDG ID
    return std::make_tuple(parent->id(), parent->pdgId());
  };

  double HPGeTree::parentTime(const art::Event& event, key_type parentId) {
    // Get the data products from the event
    auto const& StepPointMCs = event.getProduct(StepPointMCsToken);
    auto const& SimParticles = event.getProduct(SimParticlemvToken); // TODO - resture this so we don't access the SimParticles from the data product directly, but through the parent particle

    // Set up a variable to track the maximum particle time
    time = std::numeric_limits<double>::max();

    // Loop over the StepPointMCs
    for (const StepPointMC& Step : StepPointMCs) {
      // Get the step time
      stepParticle = SimParticles.at(Step.trackId());
      // If the step ID (track ID) is equal to the topmost parent and its time is lower than those previously selected, update the time
      if ((stepParticle.id() == parentId) && (Step.time() < time))
          time = Step.time();
    };

    // If the time has not been updated, throw, otherwise return the updated time
    if (time > std::numeric_limits<double>::max() * 0.99)
      throw cet::exception("LogicError") << "Time has not been updated! \n";
    return time;
  };

  void HPGeTree::analyze(const art::Event& event) {
    // Get the data products from the event
    auto const& StepPointMCs = event.getProduct(StepPointMCsToken);
    auto const& SimParticles = event.getProduct(SimParticlemvToken);

    // Validate that these data products exist
    if ((StepPointMCs.size() == 0) || (SimParticles.size() == 0))
      return;

    // Collect the particle IDs for StepPointMCs in STMDet
    // Note - IDs are from the StepPoints as the SimParticle contains the full geneaology, this is required to keep the volume constrianed to STMDet
    for (const StepPointMC& step : StepPointMCs)
      SimParticleIds.insert(SimParticles.at(step.trackId()).id());

    // Loop over all steps
    for (const StepPointMC& step : StepPointMCs) {
      // Select the appropriate steps
      if ((detector == "HPGe") && (step.position().x() > xBeamCentre))
        continue;
      else if ((detector == "LaBr") && (step.position().x() < xBeamCentre))
        continue;

      // Use the POST-step point to decide where this step's energy landed.
      // StepPointMC::position() is the PRE-step point (Mu2eG4SensitiveDetector
      // fills it from GetPreStepPoint()), which for a gamma can be a mean free
      // path away from where the energy is actually deposited. A photoabsorption
      // deposits the shell binding-energy residual locally at the post-step
      // point; testing the pre-step point dropped that deposit from ECrystal
      // whenever the gamma's previous vertex lay outside the crystal cylinder,
      // while the photoelectron (a separate track, starting inside) was still
      // counted. That produced satellite peaks exactly one Ge shell binding
      // energy below the full-energy peak (K 11.103, L 1.22-1.41, M ~0.14 keV),
      // displacing ~53% of full-energy events. See
      // analysis/HPGeWaveformStudy/spectrum/devlogs/2026-08-05_satellite_analysis_debug.md
      const CLHEP::Hep3Vector& worldPos = step.postPosition();
      const double stepEdep = step.ionizingEdep();
      const bool inCrystal = (detector == "HPGe") ? stepInCrystal(worldPos) : false;

      if (hStepXZ_half) hStepXZ_half->Fill(worldPos.x(), worldPos.z());
      if (inCrystal && hStepXZ_crystal) hStepXZ_crystal->Fill(worldPos.x(), worldPos.z());

      // Get the associated top particle
      stepParticle = SimParticles.at(step.trackId());
      std::tie(topParentId, pdgId) = topParent(SimParticleIds, stepParticle);
      // Precautionary check if the SimParticle ID is in STMDet
      if (std::find(SimParticleIds.begin(), SimParticleIds.end(), topParentId) == SimParticleIds.end())
        throw cet::exception("LogicError") << "The found parent ID is not a member of the data product\n";

      // Collate the data
      topParentIdsIt = std::find(topParentIds.begin(), topParentIds.end(), topParentId);
      if (topParentIdsIt != topParentIds.end()) {
        if (detector == "HPGe") {
          EDepsAllHalf[topParentId] += stepEdep;
          if (inCrystal) {
            EDepsCrystal[topParentId] += stepEdep;
            inCrystalFlag[topParentId] = true;
          }
        }
        // TODO - include the LaBr response here
      }
      else {
        topParentIds.emplace_back(topParentId);
        if (detector == "HPGe") {
          EDepsAllHalf.emplace(topParentId, stepEdep);
          EDepsCrystal.emplace(topParentId, inCrystal ? stepEdep : 0.0);
          inCrystalFlag.emplace(topParentId, inCrystal);
        }
        // TODO - include the LaBr response here
        times.emplace(std::make_pair(topParentId, parentTime(event, topParentId)));
      };

      // Generate the data summary
      pdgIdsIt = pdgIds.find(pdgId);
      if (pdgIdsIt != pdgIds.end())
        pdgIds[pdgId] += 1;
      else
        pdgIds.emplace(std::make_pair(pdgId, 1));
    }; // end for step

    // Collect the data to the TTrees
    for (size_t i = 0; i < topParentIds.size(); i++) {
      topParentId = topParentIds[i];
      E = EDepsAllHalf[topParentId];
      time = times[topParentId];
      ttree->Fill();
      if (inCrystalFlag[topParentId]) {
        ECrystal = EDepsCrystal[topParentId];
        ttreeCrystal->Fill();
      }
    };

    // Set up the data products for collection from the next event
    SimParticleIds.clear();
    EDepsAllHalf.clear();
    EDepsCrystal.clear();
    inCrystalFlag.clear();
    times.clear();
    topParentIds.clear();

    return;
  }; // end analyze

  void HPGeTree::endJob() {
    mf::LogInfo log("Detector tree");
    log << "==========Data summary==========\n";
    for (auto part : pdgIds)
      log << "PDGID " << part.first << ": " << part.second << "\n";
    log << "================================\n";
  };
}; // end namespace mu2e

DEFINE_ART_MODULE(mu2e::HPGeTree)
