extern "C" { 

  void pythia_init_(char input[500]) {}
  void pythia_init_default_() {}
  void pythia_setevent_() {}
  void pythia_next_() {}
  void pythia_get_stopping_info_( double scales [100][100], double mass [100][100] ) {}
  void pythia_get_dead_zones_( bool dzone [100][100] ) {}
  void pythia_clear_() {}

  void dire_init_(char input[500]) {}
  void dire_init_default_() {}
  void dire_setevent_() {}
  void dire_next_() {}
  void dire_get_stopping_info_( double scales [100][100], double mass [100][100] ) {}
  void dire_get_dead_zones_( bool dzone [100][100] ) {}
  void dire_clear_() {}


}

