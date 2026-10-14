#ifndef Mu2eG4_constructCosmicStopStand_hh
#define Mu2eG4_constructCosmicStopStand_hh
//
// Free function to construct the cosmic-muon stop stand: an absorber block
// plus scintillator paddles, all placed in one (unrotated) mother volume.
//
#include "Offline/Mu2eG4Helper/inc/VolumeInfo.hh"

namespace mu2e {

  class SimpleConfig;

  void constructCosmicStopStand( VolumeInfo   const& mother,
                                 SimpleConfig const& config );

}  // end namespace mu2e

#endif /* Mu2eG4_constructCosmicStopStand_hh */
