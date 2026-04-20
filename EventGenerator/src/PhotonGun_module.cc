// Generates photons aimed at a target (e.g. the HPGe crystal front face).
// The source position can be pulled back from the nominal (x,y,z) along the
// aim direction by `standoff` mm, and momentum is sampled uniformly inside a
// cone of half-angle `coneHalfAngle` (degrees) around the aim direction.
// Original author: Claudia Alvarez-Garcia
// Adapted by: Pawel Plesniak

// stdlib includes
#include <cmath>

// art includes
#include "art/Framework/Core/EDProducer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"

// exception handling
#include "cetlib_except/exception.h"

// fhicl includes
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/OptionalAtom.h"

// message handling
#include "messagefacility/MessageLogger/MessageLogger.h"

// Offline includes
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/DataProducts/inc/PDGCode.hh"
#include "Offline/SeedService/inc/SeedService.hh"

// CLHEP includes
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Units/PhysicalConstants.h"

namespace mu2e {
  class PhotonGun : public art::EDProducer {
  public:
    using Name=fhicl::Name;
    using Comment=fhicl::Comment;
    struct Config {
      fhicl::Atom<double> x{ Name("x"), Comment("x position of generated photon [mm]")};
      fhicl::Atom<double> y{ Name("y"), Comment("y position of generated photon [mm]")};
      fhicl::Atom<double> z{ Name("z"), Comment("z position of generated photon [mm]")};
      fhicl::Atom<double> E{ Name("E"), Comment("Energy of generated photon [MeV]")};
      fhicl::OptionalAtom<double> aimX{ Name("aimX"), Comment("Aim direction x component (unnormalized). Defaults to HPGe crystal axis sin(45 deg)")};
      fhicl::OptionalAtom<double> aimY{ Name("aimY"), Comment("Aim direction y component (unnormalized). Defaults to 0")};
      fhicl::OptionalAtom<double> aimZ{ Name("aimZ"), Comment("Aim direction z component (unnormalized). Defaults to HPGe crystal axis cos(45 deg)")};
      fhicl::OptionalAtom<double> coneHalfAngle{ Name("coneHalfAngle"), Comment("Half-angle of emission cone around aim direction [deg]. 0 = pencil beam")};
      fhicl::OptionalAtom<double> standoff{ Name("standoff"), Comment("Distance to pull the source back from (x,y,z) along -aim [mm]. Use to place the source just outside a target face")};
      fhicl::OptionalAtom<double> poissonMean{ Name("poissonMean"), Comment("Mean number of photons per event drawn from a Poisson distribution. If omitted, exactly one photon is fired per event. Set to e.g. 0.0848 for ~50 kHz at 590 kHz microspill rate.")};
    };
    using Parameters = art::EDProducer::Table<Config>;
    explicit PhotonGun(const Parameters& conf);
    virtual void produce(art::Event& event);
  private:
    CLHEP::Hep3Vector pos_;
    CLHEP::Hep3Vector aim_;
    double E_ = 0.0;
    double cosHalfAngle_ = 1.0;
    double poissonMean_ = -1.0;  // <0 means fire exactly one photon per event
    art::RandomNumberGenerator::base_engine_t& eng_;
    CLHEP::RandFlat randFlat_;
    std::unique_ptr<CLHEP::RandPoissonQ> randPoisson_;
  };

  PhotonGun::PhotonGun(const Parameters& conf):
    art::EDProducer(conf),
    E_(conf().E()),
    eng_(createEngine(art::ServiceHandle<SeedService>()->getSeed())),
    randFlat_(eng_) {
      produces<GenParticleCollection>();

      const double ax = conf().aimX() ? *conf().aimX() : std::sin(45.0*CLHEP::degree);
      const double ay = conf().aimY() ? *conf().aimY() : 0.0;
      const double az = conf().aimZ() ? *conf().aimZ() : std::cos(45.0*CLHEP::degree);
      const double amag = std::sqrt(ax*ax + ay*ay + az*az);
      if (amag < std::numeric_limits<double>::epsilon())
        throw cet::exception("RANGE") << "PhotonGun: aim direction has zero magnitude\n";
      aim_.set(ax/amag, ay/amag, az/amag);

      const double standoff = conf().standoff() ? *conf().standoff() : 0.0;
      pos_.set(conf().x() - standoff*aim_.x(),
               conf().y() - standoff*aim_.y(),
               conf().z() - standoff*aim_.z());

      const double halfAngleDeg = conf().coneHalfAngle() ? *conf().coneHalfAngle() : 0.0;
      if (halfAngleDeg < 0.0 || halfAngleDeg > 180.0)
        throw cet::exception("RANGE") << "PhotonGun: coneHalfAngle must be in [0, 180] deg\n";
      cosHalfAngle_ = std::cos(halfAngleDeg*CLHEP::degree);

      if (conf().poissonMean()) {
        poissonMean_ = *conf().poissonMean();
        if (poissonMean_ <= 0.0)
          throw cet::exception("RANGE") << "PhotonGun: poissonMean must be > 0\n";
        randPoisson_ = std::make_unique<CLHEP::RandPoissonQ>(eng_, poissonMean_);
      }
    };

  void PhotonGun::produce(art::Event& event) {
    const long nPhotons = (randPoisson_) ? randPoisson_->fire() : 1;

    std::unique_ptr<GenParticleCollection> output(new GenParticleCollection);
    for (long i = 0; i < nPhotons; ++i) {
      // Sample a direction uniformly inside a cone around aim_.
      // cos(theta) uniform in [cosHalfAngle_, 1] gives uniform solid angle.
      const double cosTheta = cosHalfAngle_ + (1.0 - cosHalfAngle_)*randFlat_.fire();
      const double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta*cosTheta));
      const double phi = CLHEP::twopi * randFlat_.fire();

      // Build an orthonormal frame (u, v, aim_) where aim_ is the +z of the local frame.
      CLHEP::Hep3Vector u, v;
      if (std::abs(aim_.x()) < 0.9)
        u = aim_.cross(CLHEP::Hep3Vector(1, 0, 0)).unit();
      else
        u = aim_.cross(CLHEP::Hep3Vector(0, 1, 0)).unit();
      v = aim_.cross(u); // already unit since aim_ and u are orthonormal

      const CLHEP::Hep3Vector dir =
        sinTheta*std::cos(phi)*u + sinTheta*std::sin(phi)*v + cosTheta*aim_;

      const CLHEP::Hep3Vector p = E_ * dir;
      CLHEP::HepLorentzVector mom(p, E_);
      output->push_back(GenParticle(PDGCode::gamma, GenId::particleGun, pos_, mom, 0.));
    }
    event.put(std::move(output));
  };
}; // end namespace mu2e

DEFINE_ART_MODULE(mu2e::PhotonGun)
