// Simulates the electronics response of the HPGe detector. Simulates the pulse height, decay tail, and ADC digitization. Generates one STMWaveformDigi per micropulse.
// Model based heavily on example provided in docDb 43617
// See docDb 51487 for full documentation
// Original author: Pawel Plesniak

// stdlib includes
#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>

// art includes
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Principal/Handle.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art/Framework/Services/Optional/RandomNumberGenerator.h"

// CLHEP includes
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Random/RandGaussQ.h"

// exception handling
#include "cetlib_except/exception.h"

// fhicl includes
#include "canvas/Utilities/InputTag.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/OptionalAtom.h"

// message handling
#include "messagefacility/MessageLogger/MessageLogger.h"

// Offline includes
// #include "Offline/DataProducts/inc/STMChannel.hh"
#include "Offline/GeometryService/inc/GeomHandle.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"
// #include "Offline/Mu2eUtilities/inc/STMUtils.hh"
// #include "Offline/ProditionsService/inc/ProditionsHandle.hh"
#include "Offline/RecoDataProducts/inc/STMWaveformDigi.hh"
#include "Offline/SeedService/inc/SeedService.hh"
#include "Offline/STMGeom/inc/HPGeDetector.hh"
#include "Offline/STMGeom/inc/STM.hh"
// #include "Offline/STMConditions/inc/STMEnergyCalib.hh"

// ROOT includes
#include "art_root_io/TFileService.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TTree.h"


namespace mu2e {
  class HPGeWaveformsFromStepPointMCs : public art::EDProducer {
  public:
    using Name=fhicl::Name;
    using Comment=fhicl::Comment;
    struct Config {
      fhicl::OptionalAtom<art::InputTag> StepPointMCsTagEle{ Name("StepPointMCsTagEle"), Comment("InputTag for StepPointMCs derived from EleBeamCat")};
      fhicl::OptionalAtom<art::InputTag> StepPointMCsTagMu{ Name("StepPointMCsTagMu"), Comment("InputTag for StepPointMCs derived from MuBeamCat")};
      fhicl::OptionalAtom<art::InputTag> StepPointMCsTag1809{ Name("StepPointMCsTag1809"), Comment("InputTag for StepPointMCs derived from TargetStopsCat")};
      fhicl::OptionalAtom<art::InputTag> StepPointMCsTag{ Name("StepPointMCsTag"), Comment("InputTag for StepPointMCs from an arbitrary source (e.g. photon gun)")};

      fhicl::Atom<double> fADC{ Name("fADC"), Comment("ADC operating frequency [MHz}")};
      fhicl::Atom<double> ADCToEnergy {Name("EnergyPerADCBin"), Comment("ADC energy calibration [keV/bin]")};
      fhicl::Atom<double> noiseSD {Name("NoiseSD"), Comment("Standard deviation of ADC noise [mV]. Set this to 0.0 for the ideal case.")};
      fhicl::Atom<double> risingEdgeDecayConstant{ Name("risingEdgeDecayConstant"), Comment("Rising edge decay time [us]")};
      fhicl::OptionalAtom<int> microspillBufferLengthCount{ Name("microspillBufferLengthCount"), Comment("Number of microspills to buffer ahead for, in number of microspills")};
      fhicl::OptionalAtom<bool> makeTTree{ Name("makeTTree"), Comment("Controls whether to make the TTree with branches chargeCollected, chargeDecayed, ADC, eventId, time")};
      fhicl::OptionalAtom<double> timeOffset{ Name("timeOffset"), Comment("For debugging, adds the named time offset in [ns], used for testing analysis algorithms")};
      fhicl::OptionalAtom<uint> resetEventNumber{ Name("resetEventNumber"), Comment("Simulates the off-spill period by resetting the inter-event last decayed charge to zero")};
      fhicl::OptionalAtom<uint> verbosityLevel{ Name("verbosityLevel"), Comment("Controls verbosity")};
    };
    using Parameters = art::EDProducer::Table<Config>;
    explicit HPGeWaveformsFromStepPointMCs(const Parameters& conf);
  private:
    void produce(art::Event& event) override;
    void beginJob();
    void beginRun(art::Run& run) override;
    void endJob() override;
    void depositCharge(const StepPointMC& step);
    void decayCharge();
    void addNoise();
    void digitize();

    // fhicl variables
    std::vector<art::InputTag> StepPointMCsTags;  // All input tags to read StepPointMCs from
    double fADC = 0;                                                                                            // ADC sampling frequency [MHz]
    double ADCToEnergy = 0;                                                                                     // Calibration of bin width to energy [keV/bin]
    double noiseSD = 0;                                                                                         // Standard deviation of ADC noise [mV]
    double risingEdgeDecayConstant = 0;                                                                         // [us]
    bool makeTTree = false;                                                                                     // Controls whether an analysis TTree is made
    double timeOffset = 0.0;                                                                                    // Used for debugging [ns]
    uint resetEventNumber = 0;                                                                                  // Event ID at which to reset the charge amplitude
    int verbosityLevel = 0;                                                                                     // How much output to generate

    // Define experiment specific constants
    const double feedbackCapacitance = 1e-12;                                                                   // [Farads]
    const double epsilonGe = 2.96;                                                                              // Energy required to generate an eh pair in Ge at 77K [eV]
    const double micropulseTime = 1695.0;                                                                       // [ns]

    // Define physics constants
    const double _e = 1.602176634e-19;                                                                          // Electric charge constant [Coulombs]
    const double electronDriftVelocity = 0.08;                                                                  // Apprixmate charged particle drift velocity [mm/ns]
    const double holeDriftVelocity = 0.06;                                                                      // Apprixmate charged particle drift velocity [mm/ns]

    // ADC variables
    double chargeToADC = 0;                                                                                     // Conversion factor from charge built in capacitor to ADC determined voltage, multiply by this value to get from charge built to ADC voltage.
    uint nADCs = 0;                                                                                             // Number of ADC values in an event
    const int16_t ADCMax = static_cast<int16_t>((-1 * std::pow(2, 15)) + 1);                                    // Maximum ADC value, power is 15 not 16 as using int16_t not uint16_t
    double ADC = 0;                                                                                             // iterator variable
    uint32_t eventTimeBuffer = 0;                                                                               // Multiple of event ids to store

    // Define Ge crystal properties [mm]
    // Crystal centre in Mu2e coords, derived from constructSTM.cc printout (2026-04-20).
    // Previous values (-3973.81, 0, 40699.1) were ~87 mm off in z and 12 mm off in x,
    // causing 99% of in-crystal steps to miss the envelope check.
    const double crystalCentreX = -3986.30;                                                                     // Crystal centre x position [mm]
    const double crystalCentreY = 0;                                                                            // Crystal centre y position [mm]
    const double crystalCentreZ = 40612.70;                                                                     // Crystal centre z position [mm]
    CLHEP::Hep3Vector crystalCentrePosition;                                                                    // Crystal centre position vector
    // TODO - want to initialize hpgeEndcapCenterPosition and holeHemisphereCenter as consts here, but errors thrown
    CLHEP::Hep3Vector hitPosition;
    const double crystalL = 78.5;                                                                               // Crystal length [mm]
    const double crystalR = 36.05;                                                                              // Crystal radius [mm]
    const double crystalHoleL = 64.7;                                                                           // Crystal hole length not including the hemisphere [mm]
    const double crystalHoleR = 5.25;                                                                           // Crystal hole radius [mm]
    const double crystalHoleZStart = crystalL - crystalHoleL;                                                   // Starting z position of the crystal hole not including the hemisphere [mm]
    const double crystalDirectionGradientCutoff = -crystalHoleZStart/crystalR;                                  // Defines a cone under which points travel to the endcap and not the curved cylinder surface
    const double stepPositionTolerance = 0.1;                                                                   // Adjusts for resolution of applying rotation
    const double maxR = crystalR + stepPositionTolerance;                                                       // Crystal radius including tolderance
    const double maxZ = crystalL + stepPositionTolerance;                                                       // Crystal z including tolerance

    // Modelling variables
    double hitR = 0;                                                                                            // Hit radial distance [mm]
    double hitZ = 0;                                                                                            // Hit axial distance [mm]
    double R0 = 0;                                                                                              // Hit radial position [mm]
    const double R1 = crystalHoleR;                                                                             // Distance travelled by electrons [mm]
    double R2 = 0;                                                                                              // Distance travelled by holes [mm]

    double trigFactor = 0;                                                                                      // Dimensionless constant used for caluclating distance
    int32_t N_ehPairs = 0;                                                                                      // Number of electron hole pairs
    uint32_t eventTime = 0;                                                                                     // Time stamp to add to STMWaveformDigi [ADC clock ticks]

    double tADC = 0;                                                                                            // Time step used for simulating the ADC values [ns]
    double electronTravelDistance = 0, holeTravelDistance = 0;                                                  // Drift distances [mm]
    double electronTravelTime = 0, holeTravelTime = 0;                                                          // Drift times [ns]
    uint32_t electronTravelTimeSteps = 0, holeTravelTimeSteps = 0;                                              // Drift times [steps]
    double decayExp = 0;                                                                                        // Amount of decay with each tADC
    double lastEventEndDecayedCharge = 0;                                                                       // Carry over for starting new microspill waveforms [charge carrier pairs]
    int microspillBufferLengthCount = 0;                                                                        // Buffer to store the charge deposits that are allocated to this event but happen after the microspill ends e.g. 844keV
    const int defaultMicrospillBufferLengthCount = 2;                                                           // Default value for the microspill buffer length

    // TTree and storage variables
    TTree* ttree;                                                                                               // ttree variable
    double chargeCollected = 0, chargeDecayed = 0;                                                              // used to fill the ttree
    uint eventId = 0;                                                                                           // used to fill the ttree
    uint32_t time = 0;                                                                                          // used to fill the ttree
    int16_t ttreeADC = 0;                                                                                       // used to fill the ttree

    // Diagnostic histograms of step world positions, filled in depositCharge
    TH2D* hStepXZ_pass = nullptr;     // steps that pass crystal-envelope bounds
    TH2D* hStepXZ_reject = nullptr;   // steps rejected by bounds
    TH1D* hStepR_pass = nullptr;      // crystal-local R, passing
    TH1D* hStepR_reject = nullptr;    // crystal-local R, rejected
    TH1D* hStepZ_pass = nullptr;      // crystal-local Z, passing
    TH1D* hStepZ_reject = nullptr;    // crystal-local Z, rejected

    // Diagnostic counters (written to stdout at endJob)
    uint64_t n_events_seen = 0;
    uint64_t n_steps_total = 0;             // all StepPointMCs seen
    uint64_t n_steps_zero_edep = 0;         // ionizingEdep == 0, skipped before depositCharge
    uint64_t n_steps_to_deposit = 0;        // reached depositCharge
    uint64_t n_reject_xcut = 0;             // hitPosition.x() > -3904 (LaBr side)
    uint64_t n_reject_timecut = 0;          // step.time() > buffer window
    uint64_t n_reject_bounds = 0;           // hitZ/hitR outside crystal envelope
    uint64_t n_deposit_ok = 0;              // charge actually deposited

    // Data storage vectors
    std::vector<double> _charge;                                                                                // Buffer to store charge collected from STMDet StepPointMCs
    std::vector<double> _chargeCollected;                                                                       // Buffer to store charge collected from STMDet StepPointMCs in the given time step
    std::vector<double> _chargeDecayed;                                                                         // Buffer to store charge collected that decays over time
    std::vector<double> _chargeCarryOver;                                                                       // Temporary buffer that will store _chargeCollected over the course of the next event
    std::vector<int16_t> _adcs;                                                                                 // Buffer for storing the ADC values to put into the STMWaveformDigi

    // Random engine + Gaussian noise distribution, seeded via SeedService so
    // every microspill draws from a distinct part of the stream (previously
    // a default-seeded engine was constructed in addNoise() each call, giving
    // identical noise on every microspill).
    art::RandomNumberGenerator::base_engine_t& _engine;
    CLHEP::RandGaussQ _noiseGauss;

    // Offline utilities
    // TODO: include the prodition to get the sampling frequency
    // mu2e::STMChannel::enum_type _HPGeChannel = static_cast<mu2e::STMChannel::enum_type>(1);
    // STMChannel* _channel = new STMChannel(_HPGeChannel);
    // ProditionsHandle<STMEnergyCalib> _stmEnergyCalib_h;
  };

  HPGeWaveformsFromStepPointMCs::HPGeWaveformsFromStepPointMCs(const Parameters& conf)
    : art::EDProducer{conf},
      fADC(conf().fADC()),
      ADCToEnergy(conf().ADCToEnergy()),
      noiseSD(conf().noiseSD()),
      risingEdgeDecayConstant(conf().risingEdgeDecayConstant()),
      _engine(createEngine(art::ServiceHandle<SeedService>()->getSeed())),
      _noiseGauss(_engine, 0.0, 1.0) {
        produces<STMWaveformDigiCollection>("HPGe");
        // Collect all configured input tags
        art::InputTag tag;
        if (conf().StepPointMCsTagEle(tag)) { mayConsume<StepPointMCCollection>(tag); StepPointMCsTags.push_back(tag); }
        if (conf().StepPointMCsTagMu(tag))  { mayConsume<StepPointMCCollection>(tag); StepPointMCsTags.push_back(tag); }
        if (conf().StepPointMCsTag1809(tag)){ mayConsume<StepPointMCCollection>(tag); StepPointMCsTags.push_back(tag); }
        if (conf().StepPointMCsTag(tag))    { mayConsume<StepPointMCCollection>(tag); StepPointMCsTags.push_back(tag); }
        if (defaultMicrospillBufferLengthCount < 2)
          throw cet::exception("RANGE", "defaultMicrospillBufferLengthCount has to be more than 1\n");

        crystalCentrePosition.set(crystalCentreX, crystalCentreY, crystalCentreZ);
        tADC = 1e3/fADC; // 1e3 converts [us] to [ns] as fADC is in [MHz]

        // Assign optional variables
        microspillBufferLengthCount = conf().microspillBufferLengthCount() ? *(conf().microspillBufferLengthCount()) : defaultMicrospillBufferLengthCount;
        verbosityLevel = conf().verbosityLevel() ? *(conf().verbosityLevel()) : 0;

        // Determine the number of ADC values in each STMWaveformDigi. Increase the number by one due to truncation. At 320MHz, this will be 543 ADC values per microbunch
        double _nADCs = (micropulseTime/tADC) + 1;
        nADCs = (int) _nADCs;
        _charge.insert(_charge.begin(), nADCs * microspillBufferLengthCount, 0.);
        _chargeCollected.insert(_chargeCollected.begin(), nADCs * microspillBufferLengthCount, 0.);
        _chargeCarryOver.insert(_chargeCarryOver.begin(), nADCs * (microspillBufferLengthCount - 1), 0.);
        _chargeDecayed.insert(_chargeDecayed.begin(), nADCs, 0.);
        _adcs.insert(_adcs.begin(), nADCs, 0);

        // Define physics parameters to use with the model
        // Define the decay amount with each step. 1e3 converts [us] to [ns]
        decayExp = exp(-tADC/(risingEdgeDecayConstant*1e3));

        // Convert noise SD from [mV] to [C], division by _e to work in units of charge
        noiseSD *= (1e-3 * feedbackCapacitance/_e);

        // 1e3 converts keV to eV, multiply by this value to get from charge in capacitor to ADC output voltage
        // Chosen to work in units of fundamental charge _e to avoid storing very small double values
        chargeToADC = epsilonGe / (ADCToEnergy * 1e3);

        // Assign the approrpiate time offset
        timeOffset = conf().timeOffset() ? *(conf().timeOffset()) : 0.0;

        // Assign TTrees
        makeTTree = conf().makeTTree() ? *(conf().makeTTree()) : false;
        if (makeTTree) {
          art::ServiceHandle<art::TFileService> tfs;
          ttree = tfs->make<TTree>("ttree", "MakeHPGeWaveformsFromStepPointMCs ttree");
          ttree->Branch("chargeCollected", &chargeCollected, "chargeCollected/D");
          ttree->Branch("chargeDecayed", &chargeDecayed, "chargeDecayed/D");
          ttree->Branch("ADC", &ttreeADC, "ADC/S");
          ttree->Branch("eventId", &eventId, "eventId/i");
          ttree->Branch("time", &time, "time/i");
        };

        resetEventNumber = conf().resetEventNumber() ? *(conf().resetEventNumber()) : 0;

        // Diagnostic histograms of step world positions, always booked
        art::ServiceHandle<art::TFileService> tfs;
        art::TFileDirectory diag = tfs->mkdir("HPGeDigiDiag");
        hStepXZ_pass   = diag.make<TH2D>("hStepXZ_pass",
                                         "Step world (x,z) passing crystal bounds;x [mm];z [mm]",
                                         200, -4050, -3850, 400, 40550, 40850);
        hStepXZ_reject = diag.make<TH2D>("hStepXZ_reject",
                                         "Step world (x,z) rejected by crystal bounds;x [mm];z [mm]",
                                         200, -4050, -3850, 400, 40550, 40850);
        hStepR_pass    = diag.make<TH1D>("hStepR_pass",
                                         "Crystal-local R (pass);R [mm];steps", 200, 0, 100);
        hStepR_reject  = diag.make<TH1D>("hStepR_reject",
                                         "Crystal-local R (reject);R [mm];steps", 200, 0, 200);
        hStepZ_pass    = diag.make<TH1D>("hStepZ_pass",
                                         "Crystal-local Z (pass);Z [mm];steps", 200, -20, 100);
        hStepZ_reject  = diag.make<TH1D>("hStepZ_reject",
                                         "Crystal-local Z (reject);Z [mm];steps", 400, -200, 200);
      };

  void HPGeWaveformsFromStepPointMCs::beginJob() {
    if (verbosityLevel) {
      std::cout << "STM HPGe digitization parameters" << std::endl;
      std::cout << "\tInput parameters" << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "fAD [MHz]"                            << fADC                                     << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "EnergyPerADCBin [keV/bin]"            << ADCToEnergy                              << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "NoiseSD [mV]"                         << noiseSD /(1e-3 * feedbackCapacitance/_e) << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "risingEdgeDecayConstant [us]"         << risingEdgeDecayConstant                  << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "microspillBufferLengthCount"          << microspillBufferLengthCount              << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "makeTTree"                            << makeTTree                                << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "timeOffset [ns]"                      << timeOffset                               << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "resetEventNumber"                     << resetEventNumber                         << std::endl;
      std::cout << "\tDerived parameters: " << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "tADC [ns]"                            << tADC                                     << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "nADCs"                                << nADCs                                    << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "NoiseSD [charge carriers]"            << noiseSD                                  << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "chargeToADC [bin/charge carrier]"     << chargeToADC                              << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "Voltage range [V]"                    << "[+1, -1]"                               << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "Voltage range used [V]"               << "[0, -1]"                                << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "Voltage range used [bins]"            << "[0, " << ADCMax << "]"                  << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "Voltage range used [charge carriers]" << "[0, " << ADCMax/chargeToADC << "]"      << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "Voltage range used [C]"               << "[0, " << ADCMax * _e/chargeToADC << "]" << std::endl;
      std::cout << std::left << "\t\t" << std::setw(60) << "Energy range [keV]"                   << "[0, " << -1*ADCMax * ADCToEnergy << "]" << std::endl;
      std::cout << std::endl; // buffer line
    };
  };

  void HPGeWaveformsFromStepPointMCs::beginRun(art::Run&) {
    // Query the actual crystal placement from GeometryService so we can compare
    // to the hard-coded crystalCentreX/Y/Z constants. If these disagree, the
    // crystal-local R/Z bounds check in depositCharge will be wrong.
    try {
      GeomHandle<STM> stm;
      HPGeDetector const* hpge = stm->getHPGeDetectorPtr();
      if (hpge) {
        CLHEP::Hep3Vector actual = hpge->originInMu2e();
        CLHEP::HepRotation const& rot = hpge->rotation();
        // NOTE: HPGeDetector::originInMu2e() is the endcap/envelope origin,
        // NOT the crystal centre. The crystal centre is offset along the
        // rotated +z axis by (WindowD + AirD + CapsuleWindowthick + CrystalL/2).
        // Leaving the hardcoded crystalCentrePosition in place; using this as
        // a diagnostic only until we add a helper to HPGeDetector that returns
        // the crystal centre directly.
        std::cout << "===== HPGe geometry check =====" << std::endl;
        std::cout << "  GeomService origin (mm): (" << actual.x() << ", "
                  << actual.y() << ", " << actual.z() << ")" << std::endl;
        std::cout << "  Hardcoded    origin (mm): (" << crystalCentreX << ", "
                  << crystalCentreY << ", " << crystalCentreZ << ")" << std::endl;
        std::cout << "  Delta (GeomService - hardcoded) (mm): ("
                  << actual.x() - crystalCentreX << ", "
                  << actual.y() - crystalCentreY << ", "
                  << actual.z() - crystalCentreZ << ")" << std::endl;
        std::cout << "  GeomService rotation matrix:" << std::endl;
        std::cout << "    [" << rot.xx() << ", " << rot.xy() << ", " << rot.xz() << "]" << std::endl;
        std::cout << "    [" << rot.yx() << ", " << rot.yy() << ", " << rot.yz() << "]" << std::endl;
        std::cout << "    [" << rot.zx() << ", " << rot.zy() << ", " << rot.zz() << "]" << std::endl;
        std::cout << "  GeomService CrystalL, CrystalR (mm): "
                  << hpge->CrystalL() << ", " << hpge->CrystalR() << std::endl;
        std::cout << "  Hardcoded    crystalL, crystalR (mm): "
                  << crystalL << ", " << crystalR << std::endl;
        std::cout << "===============================" << std::endl;
      } else {
        std::cout << "HPGe geometry check: getHPGeDetectorPtr returned null" << std::endl;
      }
    } catch (std::exception const& e) {
      std::cout << "HPGe geometry check failed: " << e.what() << std::endl;
    }
  };

  void HPGeWaveformsFromStepPointMCs::produce(art::Event& event) {
    eventId = event.id().event();
    ++n_events_seen;
    uint64_t steps_this_event = 0, deposit_this_event = 0;
    // Get the hits from all configured input tags and deposit charge
    for (const auto& tag : StepPointMCsTags) {
      art::Handle<StepPointMCCollection> handle;
      event.getByLabel(tag, handle);
      if (!handle.isValid()) continue;
      for (const StepPointMC& step : *handle) {
        ++n_steps_total;
        ++steps_this_event;
        if (step.ionizingEdep() != 0) {
          ++n_steps_to_deposit;
          uint64_t before_ok = n_deposit_ok;
          depositCharge(step);
          if (n_deposit_ok > before_ok) ++deposit_this_event;
        } else {
          ++n_steps_zero_edep;
        }
      }
    }
    if (verbosityLevel > 1 && steps_this_event > 0) {
      std::cout << "HPGeDigi event " << eventId
                << " steps=" << steps_this_event
                << " deposited=" << deposit_this_event << std::endl;
    }

    // Decay all of the collected charges
    decayCharge();

    // Update the last event decayed charge before the noise so the noise isn't added twice
    if (resetEventNumber != 0 && eventId == resetEventNumber) {
      lastEventEndDecayedCharge = 0;
      std::fill(_chargeCarryOver.begin(), _chargeCarryOver.end(), 0);
    }
    else
      lastEventEndDecayedCharge = _chargeDecayed.back();

    // Add preamplifier electronics noise with SD defined in noiseSD
    addNoise();

    // Digitize the waveform
    digitize();

    // Validate the digitized waveforms
    for (auto j : _adcs) {
      if (j > 1000)
        throw cet::exception("LogicError", "ADC values too high!");
    };

    // Simulation takes the POT time as t = 0, and has sequential microspills (events). The trigger time offset is not used here, left as a TODO
    // Create the STMWaveformDigi and insert all the relevant attributes
    // TODO - this only keeps accurate time if the sampling is 320MHz. Needs to be rewritten to work for other times
    eventTimeBuffer = eventId % 5;
    if (eventTimeBuffer == 0 || (eventId % 3) == 0)
      eventTime += nADCs + 1;
    else
      eventTime += nADCs;
    STMWaveformDigi _waveformDigi(eventTime, _adcs);
    std::unique_ptr<STMWaveformDigiCollection> outputDigis(new STMWaveformDigiCollection);
    outputDigis->emplace_back(_waveformDigi);

    // Make the ttree if appropriate
    if (makeTTree) {
      time = _waveformDigi.trigTimeOffset() - 1;
      for (uint i = 0; i < nADCs; i++) {
        chargeCollected = _chargeCollected[i];
        chargeDecayed = _chargeDecayed[i];
        ttreeADC = _adcs[i];
        time++;
        ttree->Fill();
      };
    };

    // Update the parameters to carry over to the next event
    _chargeCarryOver.clear();
    _chargeCarryOver.assign(_chargeCollected.begin() + nADCs, _chargeCollected.end());
    _chargeCollected.clear();
    _chargeCollected.assign(_chargeCarryOver.begin(), _chargeCarryOver.end());
    _chargeCollected.insert(_chargeCollected.end(), nADCs, 0);
    // Clear previous buffer vectors
    std::fill(_chargeDecayed.begin(), _chargeDecayed.end(), 0);
    std::fill(_adcs.begin(), _adcs.end(), 0);

    // Add the STMWaveformDigi to the event
    event.put(std::move(outputDigis), "HPGe");
    return;
  };

  void HPGeWaveformsFromStepPointMCs::depositCharge(const StepPointMC& step) {
    // Define variables that couldn't be constructed in the class constructor
    const CLHEP::Hep3Vector holeHemisphereCenter(0.0, 0.0, crystalHoleZStart); // Crystal hole position in local crystal co-ordinates

    hitPosition = step.position();
    // Only take the StepPoinMCs in the HPGe detector. STMDet is both sensitive volume of both the HPGe and LaBr.
    if (hitPosition.x() > -3904) {
      ++n_reject_xcut;
      return;
    }
    // If the time is outside the buffer time, skip it
    if (step.time() > (microspillBufferLengthCount * micropulseTime)) {
      ++n_reject_timecut;
      return;
    }

    // Tranform the co-ordinate system to be a cylinder in the +z direction
    // Shift the position to a local cylindrical co-ordinate system with the center of the crystal at the origin
    hitPosition -= crystalCentrePosition;
    // Rotate the cylinder for the axis to point in the +z direction
    hitPosition.rotateY(45.0*CLHEP::degree);
    // Shift the crystal so the front of the crystal is the start
    hitPosition.setZ(hitPosition.z() + (crystalL/2));

    // Redefine the hit direction.
    hitR = hitPosition.perp();
    hitZ = hitPosition.z();

    // Run checks
    if (hitZ < -stepPositionTolerance || hitZ > maxZ || hitR > maxR) {
      ++n_reject_bounds;
      if (hStepXZ_reject) hStepXZ_reject->Fill(step.position().x(), step.position().z());
      if (hStepR_reject)  hStepR_reject->Fill(hitR);
      if (hStepZ_reject)  hStepZ_reject->Fill(hitZ);
      if (n_reject_bounds <= 20) {
        std::cout << "HPGeDigi bounds-reject #" << n_reject_bounds
                  << " world=" << step.position()
                  << " local(R,Z)=(" << hitR << "," << hitZ << ")"
                  << " maxR=" << maxR << " maxZ=" << maxZ << std::endl;
      }
      return;
    }
    if (hStepXZ_pass) hStepXZ_pass->Fill(step.position().x(), step.position().z());
    if (hStepR_pass)  hStepR_pass->Fill(hitR);
    if (hStepZ_pass)  hStepZ_pass->Fill(hitZ);

    // Both electrons and holes will travel radially in all cases, but the model volume topology is different depending on the step point position
    if (hitZ > crystalHoleZStart) { // Volume is a cylinder
      electronTravelDistance = hitR + stepPositionTolerance - crystalHoleR; // Electrons travel to the center
      // Holes travel to the curved surface
      holeTravelDistance = crystalR + stepPositionTolerance - hitR;
      // Define variables for charge buildup time
      R0 = hitR;
      R2 = crystalR + stepPositionTolerance * 2;
    }
    else { // Volume is a sphere
      // Shift the origin to be on the crystal hole center
      hitPosition -= holeHemisphereCenter;
      // Redefine the variables
      hitR = hitPosition.perp();
      hitZ = hitPosition.z();
      R0 = hitPosition.mag();
      // Electrons travel to the surface of the hole hemisphere
      electronTravelDistance = R0 + stepPositionTolerance - crystalHoleR;

      // Hole motion is dependent on the position - either travelling to the curved crystal surface or the front of the crystal
      // Note - hole motion is modelled to always be radial. Required as field here is effectively radial.
      if (hitZ > (crystalDirectionGradientCutoff * hitR)) { // Holes travel to the curved surface
        trigFactor = hitR / R0; // trigFactor is positive solution by construction
        holeTravelDistance = (crystalR / trigFactor) - R0 + stepPositionTolerance;
      }
      else { // Holes travel to the endcap
        trigFactor = -hitZ / R0; // trigFactor is positive solution by construction
        holeTravelDistance = (crystalHoleZStart / trigFactor) - R0 + stepPositionTolerance;
      };
      R2 = electronTravelDistance + holeTravelDistance + crystalHoleR + stepPositionTolerance * 2;
    };

    if (holeTravelDistance < 0 || electronTravelDistance < 0)
      throw cet::exception("LogicError") << "Electron (" << electronTravelDistance << ") and hole (" << holeTravelDistance << ") travelling distances should both be positve.\nPosition found at " << step.position() << "(" << hitPosition << ")\n";

    // Calculate the drift times
    electronTravelTime = electronTravelDistance / electronDriftVelocity;
    holeTravelTime = holeTravelDistance / holeDriftVelocity;

    // Calcuate the number of eh pairs from the ionizing energy deposition.
    N_ehPairs = -1.0 * step.ionizingEdep() * 1e6 / epsilonGe; // 1e6 converts MeV to eV. -1.0 as this is a decreasing peak

    // Define parameters required for charge deposition. Constants A and B are defined here for code brevity
    uint tIndex = (step.time() + timeOffset) / tADC, tIndexStart = tIndex;
    const double A = N_ehPairs / log(R2/R1);
    const double Be = electronDriftVelocity / R0;
    const double Bh = holeDriftVelocity / R0;

    // Scale and shift the charge particle travel time
    electronTravelTimeSteps = tIndex + electronTravelTime/tADC;
    holeTravelTimeSteps = tIndex + holeTravelTime/tADC;

    // Calculate charge collection when both particles are moving
    while (tIndex <= electronTravelTimeSteps && tIndex <= holeTravelTimeSteps) {
      _charge[tIndex] = A*log((1 + Bh * (tIndex - tIndexStart) * tADC) / (1 - Be * (tIndex - tIndexStart) * tADC));
      tIndex++;
    };

    // Calculate charge build up when only the holes are moving
    if (tIndex >= electronTravelTimeSteps && tIndex < holeTravelTimeSteps) {
      while(tIndex < holeTravelTimeSteps) {
        _charge[tIndex] = A * log((1 + Bh * (tIndex - tIndexStart) * tADC ) / (R1 / R0));
        tIndex++;
      };
    }
    // Calculate charge build up when only the electrons are moving
    else if (tIndex < electronTravelTimeSteps && tIndex >= holeTravelTimeSteps) {
      while(tIndex < electronTravelTimeSteps) {
        _charge[tIndex] = A * log((R2 / R0) / (1 - Be * (tIndex - tIndexStart) * tADC));
        tIndex++;
      };
    };

    // Allocate the rest of the charge for one more entry for the continuity
    _charge[tIndex] = N_ehPairs;
    tIndex++;

    // Update _chargeCollected. First case is treated separately as there is no charge deposited in the previous step
    if(tIndexStart == 0) {
      _chargeCollected[tIndexStart] += _charge[tIndexStart];
      tIndexStart++;
    };
    for (uint i = tIndexStart; i < tIndex; i++)
      _chargeCollected[i] += (_charge[i] - _charge[i-1]);

    // Clear the charge vector
    std::fill(_charge.begin(), _charge.end(), 0);
    ++n_deposit_ok;
    return;
  };

  void HPGeWaveformsFromStepPointMCs::endJob() {
    std::cout << "===== HPGeWaveformsFromStepPointMCs loss breakdown =====" << std::endl;
    std::cout << "  events seen                : " << n_events_seen     << std::endl;
    std::cout << "  steps total                : " << n_steps_total     << std::endl;
    std::cout << "  steps with ionizingEdep==0 : " << n_steps_zero_edep << std::endl;
    std::cout << "  steps reaching depositCharge: " << n_steps_to_deposit << std::endl;
    std::cout << "    rejected: x > -3904 (LaBr): " << n_reject_xcut     << std::endl;
    std::cout << "    rejected: step.time > buf : " << n_reject_timecut  << std::endl;
    std::cout << "    rejected: out-of-bounds   : " << n_reject_bounds   << std::endl;
    std::cout << "  deposits successful        : " << n_deposit_ok      << std::endl;
    std::cout << "========================================================" << std::endl;
  };

  void HPGeWaveformsFromStepPointMCs::decayCharge() {
    _chargeDecayed[0] = lastEventEndDecayedCharge * decayExp + _chargeCollected[0];
    for (uint t = 1; t < nADCs; t++)
      _chargeDecayed[t] = _chargeDecayed[t-1] * decayExp + _chargeCollected[t];
    return;
  };

  void HPGeWaveformsFromStepPointMCs::addNoise() {
    // If the noise SD is zero, do nothing
    if (noiseSD < std::numeric_limits<double>::epsilon())
      return;

    // Draw Gaussian noise per sample from the SeedService-seeded engine.
    // _noiseGauss has unit variance; multiply by noiseSD to get the
    // configured noise amplitude (in the same charge-carrier units as
    // _chargeDecayed after the mV->charge conversion in the constructor).
    for (size_t _i = 0; _i < nADCs; _i++)
      _chargeDecayed[_i] += noiseSD * _noiseGauss.fire();
    return;
  };

  void HPGeWaveformsFromStepPointMCs::digitize() {
    // Convert the charge deposition to ADC voltage output.
    for (uint i = 0; i < nADCs; i++) {
      ADC = _chargeDecayed[i] * chargeToADC;
      _adcs[i] = ADC > ADCMax ? static_cast<int16_t>(std::round(ADC)) : ADCMax;
    };
    return;
  };
}; // namespace mu2e

DEFINE_ART_MODULE(mu2e::HPGeWaveformsFromStepPointMCs)
