extern "C" { 

  // an initialisation function
  void pythia_init_(char input[500]) {}
  void pythia_init_default_() {}

  // a function to shower and analyse events
  void pythia_setevent_() {}

  // a function to shower and analyse events
  void pythia_next_() {}

    //This should set the LHA event using fortran common blocks
  //a function to close everything
  void pythia_stat_() {}

  void dire_init_(char input[500]) {}
  void dire_init_default_() {}
  void dire_setevent_() {}
  void dire_next_() {}
  void dire_stat_() {}
  void dire_get_mergingweight_(double& w) {}

}

