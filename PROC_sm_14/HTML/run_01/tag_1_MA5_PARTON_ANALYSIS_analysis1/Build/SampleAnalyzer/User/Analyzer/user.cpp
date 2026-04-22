#include "SampleAnalyzer/User/Analyzer/user.h"
using namespace MA5;

MAbool user::Initialize(const MA5::Configuration& cfg,
                      const std::map<std::string,std::string>& parameters)
{
  // Initializing PhysicsService for MC
  PHYSICS->mcConfig().Reset();

  // definition of the multiparticle "hadronic"
  PHYSICS->mcConfig().AddHadronicId(-5);
  PHYSICS->mcConfig().AddHadronicId(-4);
  PHYSICS->mcConfig().AddHadronicId(-3);
  PHYSICS->mcConfig().AddHadronicId(-2);
  PHYSICS->mcConfig().AddHadronicId(-1);
  PHYSICS->mcConfig().AddHadronicId(1);
  PHYSICS->mcConfig().AddHadronicId(2);
  PHYSICS->mcConfig().AddHadronicId(3);
  PHYSICS->mcConfig().AddHadronicId(4);
  PHYSICS->mcConfig().AddHadronicId(5);
  PHYSICS->mcConfig().AddHadronicId(21);

  // definition of the multiparticle "invisible"
  PHYSICS->mcConfig().AddInvisibleId(-16);
  PHYSICS->mcConfig().AddInvisibleId(-14);
  PHYSICS->mcConfig().AddInvisibleId(-12);
  PHYSICS->mcConfig().AddInvisibleId(12);
  PHYSICS->mcConfig().AddInvisibleId(14);
  PHYSICS->mcConfig().AddInvisibleId(16);

  // ===== Signal region ===== //
  Manager()->AddRegionSelection("myregion");

  // ===== Selections ===== //

  // ===== Histograms ===== //
  Manager()->AddHisto("1_THT", 40,0.0,500.0);
  Manager()->AddHisto("2_MET", 40,0.0,500.0);
  Manager()->AddHisto("3_SQRTS", 40,0.0,500.0);
  Manager()->AddHisto("4_PT", 40,0.0,500.0);
  Manager()->AddHisto("5_ETA", 40,-10.0,10.0);
  Manager()->AddHisto("6_PT", 40,0.0,500.0);
  Manager()->AddHisto("7_ETA", 40,-10.0,10.0);
  Manager()->AddHisto("8_PT", 40,0.0,500.0);
  Manager()->AddHisto("9_ETA", 40,-10.0,10.0);
  Manager()->AddHisto("10_PT", 40,0.0,500.0);
  Manager()->AddHisto("11_ETA", 40,-10.0,10.0);
  Manager()->AddHisto("12_PT", 40,0.0,500.0);
  Manager()->AddHisto("13_ETA", 40,-10.0,10.0);
  Manager()->AddHisto("14_M", 40,0.0,500.0);
  Manager()->AddHisto("15_M", 40,0.0,500.0);
  Manager()->AddHisto("16_M", 40,0.0,500.0);
  Manager()->AddHisto("17_M", 40,0.0,500.0);
  Manager()->AddHisto("18_M", 40,0.0,500.0);
  Manager()->AddHisto("19_M", 40,0.0,500.0);
  Manager()->AddHisto("20_M", 40,0.0,500.0);
  Manager()->AddHisto("21_M", 40,0.0,500.0);
  Manager()->AddHisto("22_M", 40,0.0,500.0);
  Manager()->AddHisto("23_M", 40,0.0,500.0);
  Manager()->AddHisto("24_M", 40,0.0,500.0);
  Manager()->AddHisto("25_M", 40,0.0,500.0);
  Manager()->AddHisto("26_M", 40,0.0,500.0);
  Manager()->AddHisto("27_M", 40,0.0,500.0);
  Manager()->AddHisto("28_M", 40,0.0,500.0);
  Manager()->AddHisto("29_M", 40,0.0,500.0);
  Manager()->AddHisto("30_M", 40,0.0,500.0);
  Manager()->AddHisto("31_M", 40,0.0,500.0);
  Manager()->AddHisto("32_M", 40,0.0,500.0);
  Manager()->AddHisto("33_M", 40,0.0,500.0);
  Manager()->AddHisto("34_M", 40,0.0,500.0);
  Manager()->AddHisto("35_M", 40,0.0,500.0);
  Manager()->AddHisto("36_M", 40,0.0,500.0);
  Manager()->AddHisto("37_M", 40,0.0,500.0);
  Manager()->AddHisto("38_M", 40,0.0,500.0);
  Manager()->AddHisto("39_M", 40,0.0,500.0);
  Manager()->AddHisto("40_DELTAR", 40,0.0,10.0);
  Manager()->AddHisto("41_DELTAR", 40,0.0,10.0);
  Manager()->AddHisto("42_DELTAR", 40,0.0,10.0);
  Manager()->AddHisto("43_DELTAR", 40,0.0,10.0);
  Manager()->AddHisto("44_DELTAR", 40,0.0,10.0);
  Manager()->AddHisto("45_DELTAR", 40,0.0,10.0);
  Manager()->AddHisto("46_DELTAR", 40,0.0,10.0);
  Manager()->AddHisto("47_DELTAR", 40,0.0,10.0);
  Manager()->AddHisto("48_DELTAR", 40,0.0,10.0);
  Manager()->AddHisto("49_DELTAR", 40,0.0,10.0);

  // No problem during initialization
  return true;
}

MAbool user::Execute(SampleFormat& sample, const EventFormat& event)
{
  MAfloat32 __event_weight__ = 1.0;
  if (weighted_events_ && event.mc()!=0) __event_weight__ = event.mc()->weight();

  if (sample.mc()!=0) sample.mc()->addWeightedEvents(__event_weight__);
  Manager()->InitializeForNewEvent(__event_weight__);

  // Clearing particle containers
  {
      _P_l_p_I1I_PTorderingfinalstate_REG_.clear();
      _P_l_m_I1I_PTorderingfinalstate_REG_.clear();
      _P_p_I1I_PTorderingfinalstate_REG_.clear();
      _P_p_I2I_PTorderingfinalstate_REG_.clear();
      _P_p_I3I_PTorderingfinalstate_REG_.clear();
      _P_l_pPTorderingfinalstate_REG_.clear();
      _P_l_mPTorderingfinalstate_REG_.clear();
      _P_pPTorderingfinalstate_REG_.clear();
  }

  // Filling particle containers
  {
    for (MAuint32 i=0;i<event.mc()->particles().size();i++)
    {
      if (isP__l_pPTorderingfinalstate((&(event.mc()->particles()[i])))) _P_l_pPTorderingfinalstate_REG_.push_back(&(event.mc()->particles()[i]));
      if (isP__l_mPTorderingfinalstate((&(event.mc()->particles()[i])))) _P_l_mPTorderingfinalstate_REG_.push_back(&(event.mc()->particles()[i]));
      if (isP__pPTorderingfinalstate((&(event.mc()->particles()[i])))) _P_pPTorderingfinalstate_REG_.push_back(&(event.mc()->particles()[i]));
    }
  }

  // Sorting particles
  // Sorting particle collection according to PTordering
  // for getting 1th particle
  _P_l_p_I1I_PTorderingfinalstate_REG_=SORTER->rankFilter(_P_l_pPTorderingfinalstate_REG_,1,PTordering);

  // Sorting particle collection according to PTordering
  // for getting 1th particle
  _P_l_m_I1I_PTorderingfinalstate_REG_=SORTER->rankFilter(_P_l_mPTorderingfinalstate_REG_,1,PTordering);

  // Sorting particle collection according to PTordering
  // for getting 1th particle
  _P_p_I1I_PTorderingfinalstate_REG_=SORTER->rankFilter(_P_pPTorderingfinalstate_REG_,1,PTordering);

  // Sorting particle collection according to PTordering
  // for getting 2th particle
  _P_p_I2I_PTorderingfinalstate_REG_=SORTER->rankFilter(_P_pPTorderingfinalstate_REG_,2,PTordering);

  // Sorting particle collection according to PTordering
  // for getting 3th particle
  _P_p_I3I_PTorderingfinalstate_REG_=SORTER->rankFilter(_P_pPTorderingfinalstate_REG_,3,PTordering);

  // Histogram number 1
  //   * Plot: THT
  {
    Manager()->FillHisto("1_THT", PHYSICS->Transverse->EventTHT(event.mc()));
  }

  // Histogram number 2
  //   * Plot: MET
  {
    Manager()->FillHisto("2_MET", PHYSICS->Transverse->EventMET(event.mc()));
  }

  // Histogram number 3
  //   * Plot: SQRTS
  {
    Manager()->FillHisto("3_SQRTS", PHYSICS->SqrtS(event.mc()));
  }

  // Histogram number 4
  //   * Plot: PT ( l-[1] ) 
  {
  {
    MAuint32 ind[1];
    for (ind[0]=0;ind[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
      Manager()->FillHisto("4_PT", _P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]->pt());
    }
  }
  }

  // Histogram number 5
  //   * Plot: ETA ( l-[1] ) 
  {
  {
    MAuint32 ind[1];
    for (ind[0]=0;ind[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
      Manager()->FillHisto("5_ETA", _P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]->eta());
    }
  }
  }

  // Histogram number 6
  //   * Plot: PT ( l+[1] ) 
  {
  {
    MAuint32 ind[1];
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
      Manager()->FillHisto("6_PT", _P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->pt());
    }
  }
  }

  // Histogram number 7
  //   * Plot: ETA ( l+[1] ) 
  {
  {
    MAuint32 ind[1];
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
      Manager()->FillHisto("7_ETA", _P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->eta());
    }
  }
  }

  // Histogram number 8
  //   * Plot: PT ( p[1] ) 
  {
  {
    MAuint32 ind[1];
    for (ind[0]=0;ind[0]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
      Manager()->FillHisto("8_PT", _P_p_I1I_PTorderingfinalstate_REG_[ind[0]]->pt());
    }
  }
  }

  // Histogram number 9
  //   * Plot: ETA ( p[1] ) 
  {
  {
    MAuint32 ind[1];
    for (ind[0]=0;ind[0]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
      Manager()->FillHisto("9_ETA", _P_p_I1I_PTorderingfinalstate_REG_[ind[0]]->eta());
    }
  }
  }

  // Histogram number 10
  //   * Plot: PT ( p[2] ) 
  {
  {
    MAuint32 ind[1];
    for (ind[0]=0;ind[0]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
      Manager()->FillHisto("10_PT", _P_p_I2I_PTorderingfinalstate_REG_[ind[0]]->pt());
    }
  }
  }

  // Histogram number 11
  //   * Plot: ETA ( p[2] ) 
  {
  {
    MAuint32 ind[1];
    for (ind[0]=0;ind[0]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
      Manager()->FillHisto("11_ETA", _P_p_I2I_PTorderingfinalstate_REG_[ind[0]]->eta());
    }
  }
  }

  // Histogram number 12
  //   * Plot: PT ( p[3] ) 
  {
  {
    MAuint32 ind[1];
    for (ind[0]=0;ind[0]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
      Manager()->FillHisto("12_PT", _P_p_I3I_PTorderingfinalstate_REG_[ind[0]]->pt());
    }
  }
  }

  // Histogram number 13
  //   * Plot: ETA ( p[3] ) 
  {
  {
    MAuint32 ind[1];
    for (ind[0]=0;ind[0]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
      Manager()->FillHisto("13_ETA", _P_p_I3I_PTorderingfinalstate_REG_[ind[0]]->eta());
    }
  }
  }

  // Histogram number 14
  //   * Plot: M ( l+[1] p[1] ) 
  {
  {
    MAuint32 ind[2];
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
      Manager()->FillHisto("14_M", q.m());
    }
    }
  }
  }

  // Histogram number 15
  //   * Plot: M ( l+[1] p[1] p[2] ) 
  {
  {
    MAuint32 ind[3];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I2I_PTorderingfinalstate_REG_[ind[2]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<3;i++)
    {
      mycombi.insert(_P_l_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]->momentum();
      Manager()->FillHisto("15_M", q.m());
    }
    }
    }
  }
  }

  // Histogram number 16
  //   * Plot: M ( l+[1] p[1] p[2] p[3] ) 
  {
  {
    MAuint32 ind[4];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I2I_PTorderingfinalstate_REG_[ind[2]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]) continue;
    for (ind[3]=0;ind[3]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[3]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[1]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<4;i++)
    {
      mycombi.insert(_P_l_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[3]]->momentum();
      Manager()->FillHisto("16_M", q.m());
    }
    }
    }
    }
  }
  }

  // Histogram number 17
  //   * Plot: M ( l+[1] p[1] p[3] ) 
  {
  {
    MAuint32 ind[3];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[2]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[2]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<3;i++)
    {
      mycombi.insert(_P_l_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[2]]->momentum();
      Manager()->FillHisto("17_M", q.m());
    }
    }
    }
  }
  }

  // Histogram number 18
  //   * Plot: M ( l+[1] p[2] ) 
  {
  {
    MAuint32 ind[2];
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]->momentum();
      Manager()->FillHisto("18_M", q.m());
    }
    }
  }
  }

  // Histogram number 19
  //   * Plot: M ( l+[1] p[2] p[3] ) 
  {
  {
    MAuint32 ind[3];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[2]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[2]]==_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<3;i++)
    {
      mycombi.insert(_P_l_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[2]]->momentum();
      Manager()->FillHisto("19_M", q.m());
    }
    }
    }
  }
  }

  // Histogram number 20
  //   * Plot: M ( l+[1] p[3] ) 
  {
  {
    MAuint32 ind[2];
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[1]]->momentum();
      Manager()->FillHisto("20_M", q.m());
    }
    }
  }
  }

  // Histogram number 21
  //   * Plot: M ( l+[1] l-[1] ) 
  {
  {
    MAuint32 ind[2];
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
      Manager()->FillHisto("21_M", q.m());
    }
    }
  }
  }

  // Histogram number 22
  //   * Plot: M ( l+[1] l-[1] p[1] ) 
  {
  {
    MAuint32 ind[3];
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    for (ind[2]=0;ind[2]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[2]]->momentum();
      Manager()->FillHisto("22_M", q.m());
    }
    }
    }
  }
  }

  // Histogram number 23
  //   * Plot: M ( l+[1] l-[1] p[1] p[2] ) 
  {
  {
    MAuint32 ind[4];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I1I_PTorderingfinalstate_REG_[ind[2]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I1I_PTorderingfinalstate_REG_[ind[2]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]) continue;
    for (ind[3]=0;ind[3]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[3]++)
    {
    if (_P_p_I2I_PTorderingfinalstate_REG_[ind[3]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I2I_PTorderingfinalstate_REG_[ind[3]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]] || _P_p_I2I_PTorderingfinalstate_REG_[ind[3]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[2]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<4;i++)
    {
      mycombi.insert(_P_l_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_l_m_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[2]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[3]]->momentum();
      Manager()->FillHisto("23_M", q.m());
    }
    }
    }
    }
  }
  }

  // Histogram number 24
  //   * Plot: M ( l+[1] l-[1] p[1] p[2] p[3] ) 
  {
  {
    MAuint32 ind[5];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I1I_PTorderingfinalstate_REG_[ind[2]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I1I_PTorderingfinalstate_REG_[ind[2]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]) continue;
    for (ind[3]=0;ind[3]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[3]++)
    {
    if (_P_p_I2I_PTorderingfinalstate_REG_[ind[3]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I2I_PTorderingfinalstate_REG_[ind[3]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]] || _P_p_I2I_PTorderingfinalstate_REG_[ind[3]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[2]]) continue;
    for (ind[4]=0;ind[4]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[4]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[4]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[4]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[4]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[2]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[4]]==_P_p_I2I_PTorderingfinalstate_REG_[ind[3]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<5;i++)
    {
      mycombi.insert(_P_l_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_l_m_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[2]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[3]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[4]]->momentum();
      Manager()->FillHisto("24_M", q.m());
    }
    }
    }
    }
    }
  }
  }

  // Histogram number 25
  //   * Plot: M ( l+[1] l-[1] p[1] p[3] ) 
  {
  {
    MAuint32 ind[4];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I1I_PTorderingfinalstate_REG_[ind[2]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I1I_PTorderingfinalstate_REG_[ind[2]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]) continue;
    for (ind[3]=0;ind[3]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[3]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[2]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<4;i++)
    {
      mycombi.insert(_P_l_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_l_m_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[2]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[3]]->momentum();
      Manager()->FillHisto("25_M", q.m());
    }
    }
    }
    }
  }
  }

  // Histogram number 26
  //   * Plot: M ( l+[1] l-[1] p[2] ) 
  {
  {
    MAuint32 ind[3];
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    for (ind[2]=0;ind[2]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]->momentum();
      Manager()->FillHisto("26_M", q.m());
    }
    }
    }
  }
  }

  // Histogram number 27
  //   * Plot: M ( l+[1] l-[1] p[2] p[3] ) 
  {
  {
    MAuint32 ind[4];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I2I_PTorderingfinalstate_REG_[ind[2]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]) continue;
    for (ind[3]=0;ind[3]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[3]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<4;i++)
    {
      mycombi.insert(_P_l_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_l_m_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[3]]->momentum();
      Manager()->FillHisto("27_M", q.m());
    }
    }
    }
    }
  }
  }

  // Histogram number 28
  //   * Plot: M ( l+[1] l-[1] p[3] ) 
  {
  {
    MAuint32 ind[3];
    for (ind[0]=0;ind[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    for (ind[2]=0;ind[2]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    ParticleBaseFormat q;
    q+=_P_l_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[2]]->momentum();
      Manager()->FillHisto("28_M", q.m());
    }
    }
    }
  }
  }

  // Histogram number 29
  //   * Plot: M ( l-[1] p[1] ) 
  {
  {
    MAuint32 ind[2];
    for (ind[0]=0;ind[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    ParticleBaseFormat q;
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
      Manager()->FillHisto("29_M", q.m());
    }
    }
  }
  }

  // Histogram number 30
  //   * Plot: M ( l-[1] p[1] p[2] ) 
  {
  {
    MAuint32 ind[3];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I2I_PTorderingfinalstate_REG_[ind[2]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<3;i++)
    {
      mycombi.insert(_P_l_m_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]->momentum();
      Manager()->FillHisto("30_M", q.m());
    }
    }
    }
  }
  }

  // Histogram number 31
  //   * Plot: M ( l-[1] p[1] p[2] p[3] ) 
  {
  {
    MAuint32 ind[4];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I2I_PTorderingfinalstate_REG_[ind[2]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]) continue;
    for (ind[3]=0;ind[3]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[3]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[1]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[3]]==_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<4;i++)
    {
      mycombi.insert(_P_l_m_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[2]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[3]]->momentum();
      Manager()->FillHisto("31_M", q.m());
    }
    }
    }
    }
  }
  }

  // Histogram number 32
  //   * Plot: M ( l-[1] p[1] p[3] ) 
  {
  {
    MAuint32 ind[3];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[2]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[2]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<3;i++)
    {
      mycombi.insert(_P_l_m_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[2]]->momentum();
      Manager()->FillHisto("32_M", q.m());
    }
    }
    }
  }
  }

  // Histogram number 33
  //   * Plot: M ( l-[1] p[2] ) 
  {
  {
    MAuint32 ind[2];
    for (ind[0]=0;ind[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    ParticleBaseFormat q;
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]->momentum();
      Manager()->FillHisto("33_M", q.m());
    }
    }
  }
  }

  // Histogram number 34
  //   * Plot: M ( l-[1] p[2] p[3] ) 
  {
  {
    MAuint32 ind[3];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[2]]==_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[2]]==_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<3;i++)
    {
      mycombi.insert(_P_l_m_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[2]]->momentum();
      Manager()->FillHisto("34_M", q.m());
    }
    }
    }
  }
  }

  // Histogram number 35
  //   * Plot: M ( l-[1] p[3] ) 
  {
  {
    MAuint32 ind[2];
    for (ind[0]=0;ind[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    ParticleBaseFormat q;
    q+=_P_l_m_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[1]]->momentum();
      Manager()->FillHisto("35_M", q.m());
    }
    }
  }
  }

  // Histogram number 36
  //   * Plot: M ( p[1] p[2] ) 
  {
  {
    MAuint32 ind[2];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<2;i++)
    {
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]->momentum();
      Manager()->FillHisto("36_M", q.m());
    }
    }
  }
  }

  // Histogram number 37
  //   * Plot: M ( p[1] p[2] p[3] ) 
  {
  {
    MAuint32 ind[3];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;
    for (ind[2]=0;ind[2]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[2]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[2]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[0]] || _P_p_I3I_PTorderingfinalstate_REG_[ind[2]]==_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<3;i++)
    {
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[1]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[2]]->momentum();
      Manager()->FillHisto("37_M", q.m());
    }
    }
    }
  }
  }

  // Histogram number 38
  //   * Plot: M ( p[1] p[3] ) 
  {
  {
    MAuint32 ind[2];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_p_I1I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[1]]==_P_p_I1I_PTorderingfinalstate_REG_[ind[0]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<2;i++)
    {
      mycombi.insert(_P_p_I1I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_p_I1I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[1]]->momentum();
      Manager()->FillHisto("38_M", q.m());
    }
    }
  }
  }

  // Histogram number 39
  //   * Plot: M ( p[2] p[3] ) 
  {
  {
    MAuint32 ind[2];
    std::vector<std::set<const MCParticleFormat*> > combis;
    for (ind[0]=0;ind[0]<_P_p_I2I_PTorderingfinalstate_REG_.size();ind[0]++)
    {
    for (ind[1]=0;ind[1]<_P_p_I3I_PTorderingfinalstate_REG_.size();ind[1]++)
    {
    if (_P_p_I3I_PTorderingfinalstate_REG_[ind[1]]==_P_p_I2I_PTorderingfinalstate_REG_[ind[0]]) continue;

    // Checking if consistent combination
    std::set<const MCParticleFormat*> mycombi;
    for (MAuint32 i=0;i<2;i++)
    {
      mycombi.insert(_P_p_I2I_PTorderingfinalstate_REG_[ind[i]]);
      mycombi.insert(_P_p_I3I_PTorderingfinalstate_REG_[ind[i]]);
    }
    MAbool matched=false;
    for (MAuint32 i=0;i<combis.size();i++)
      if (combis[i]==mycombi) {matched=true; break;}
    if (matched) continue;
    else combis.push_back(mycombi);

    ParticleBaseFormat q;
    q+=_P_p_I2I_PTorderingfinalstate_REG_[ind[0]]->momentum();
    q+=_P_p_I3I_PTorderingfinalstate_REG_[ind[1]]->momentum();
      Manager()->FillHisto("39_M", q.m());
    }
    }
  }
  }

  // Histogram number 40
  //   * Plot: DELTAR ( l+[1] , p[1] ) 
  {
  {
    MAuint32 a[1];
    for (a[0]=0;a[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();a[0]++)
    {
    MAuint32 b[1];
    for (b[0]=0;b[0]<_P_p_I1I_PTorderingfinalstate_REG_.size();b[0]++)
    {
      Manager()->FillHisto("40_DELTAR", _P_l_p_I1I_PTorderingfinalstate_REG_[a[0]]->dr(_P_p_I1I_PTorderingfinalstate_REG_[b[0]]));
    }
    }
  }
  }

  // Histogram number 41
  //   * Plot: DELTAR ( l+[1] , p[2] ) 
  {
  {
    MAuint32 a[1];
    for (a[0]=0;a[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();a[0]++)
    {
    MAuint32 b[1];
    for (b[0]=0;b[0]<_P_p_I2I_PTorderingfinalstate_REG_.size();b[0]++)
    {
      Manager()->FillHisto("41_DELTAR", _P_l_p_I1I_PTorderingfinalstate_REG_[a[0]]->dr(_P_p_I2I_PTorderingfinalstate_REG_[b[0]]));
    }
    }
  }
  }

  // Histogram number 42
  //   * Plot: DELTAR ( l+[1] , p[3] ) 
  {
  {
    MAuint32 a[1];
    for (a[0]=0;a[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();a[0]++)
    {
    MAuint32 b[1];
    for (b[0]=0;b[0]<_P_p_I3I_PTorderingfinalstate_REG_.size();b[0]++)
    {
      Manager()->FillHisto("42_DELTAR", _P_l_p_I1I_PTorderingfinalstate_REG_[a[0]]->dr(_P_p_I3I_PTorderingfinalstate_REG_[b[0]]));
    }
    }
  }
  }

  // Histogram number 43
  //   * Plot: DELTAR ( l-[1] , l+[1] ) 
  {
  {
    MAuint32 a[1];
    for (a[0]=0;a[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();a[0]++)
    {
    MAuint32 b[1];
    for (b[0]=0;b[0]<_P_l_p_I1I_PTorderingfinalstate_REG_.size();b[0]++)
    {
      Manager()->FillHisto("43_DELTAR", _P_l_m_I1I_PTorderingfinalstate_REG_[a[0]]->dr(_P_l_p_I1I_PTorderingfinalstate_REG_[b[0]]));
    }
    }
  }
  }

  // Histogram number 44
  //   * Plot: DELTAR ( l-[1] , p[1] ) 
  {
  {
    MAuint32 a[1];
    for (a[0]=0;a[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();a[0]++)
    {
    MAuint32 b[1];
    for (b[0]=0;b[0]<_P_p_I1I_PTorderingfinalstate_REG_.size();b[0]++)
    {
      Manager()->FillHisto("44_DELTAR", _P_l_m_I1I_PTorderingfinalstate_REG_[a[0]]->dr(_P_p_I1I_PTorderingfinalstate_REG_[b[0]]));
    }
    }
  }
  }

  // Histogram number 45
  //   * Plot: DELTAR ( l-[1] , p[2] ) 
  {
  {
    MAuint32 a[1];
    for (a[0]=0;a[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();a[0]++)
    {
    MAuint32 b[1];
    for (b[0]=0;b[0]<_P_p_I2I_PTorderingfinalstate_REG_.size();b[0]++)
    {
      Manager()->FillHisto("45_DELTAR", _P_l_m_I1I_PTorderingfinalstate_REG_[a[0]]->dr(_P_p_I2I_PTorderingfinalstate_REG_[b[0]]));
    }
    }
  }
  }

  // Histogram number 46
  //   * Plot: DELTAR ( l-[1] , p[3] ) 
  {
  {
    MAuint32 a[1];
    for (a[0]=0;a[0]<_P_l_m_I1I_PTorderingfinalstate_REG_.size();a[0]++)
    {
    MAuint32 b[1];
    for (b[0]=0;b[0]<_P_p_I3I_PTorderingfinalstate_REG_.size();b[0]++)
    {
      Manager()->FillHisto("46_DELTAR", _P_l_m_I1I_PTorderingfinalstate_REG_[a[0]]->dr(_P_p_I3I_PTorderingfinalstate_REG_[b[0]]));
    }
    }
  }
  }

  // Histogram number 47
  //   * Plot: DELTAR ( p[1] , p[2] ) 
  {
  {
    MAuint32 a[1];
    for (a[0]=0;a[0]<_P_p_I1I_PTorderingfinalstate_REG_.size();a[0]++)
    {
    MAuint32 b[1];
    for (b[0]=0;b[0]<_P_p_I2I_PTorderingfinalstate_REG_.size();b[0]++)
    {
     if ( _P_p_I1I_PTorderingfinalstate_REG_[a[0]] == _P_p_I2I_PTorderingfinalstate_REG_[b[0]] ) continue;
      Manager()->FillHisto("47_DELTAR", _P_p_I1I_PTorderingfinalstate_REG_[a[0]]->dr(_P_p_I2I_PTorderingfinalstate_REG_[b[0]]));
    }
    }
  }
  }

  // Histogram number 48
  //   * Plot: DELTAR ( p[1] , p[3] ) 
  {
  {
    MAuint32 a[1];
    for (a[0]=0;a[0]<_P_p_I1I_PTorderingfinalstate_REG_.size();a[0]++)
    {
    MAuint32 b[1];
    for (b[0]=0;b[0]<_P_p_I3I_PTorderingfinalstate_REG_.size();b[0]++)
    {
     if ( _P_p_I1I_PTorderingfinalstate_REG_[a[0]] == _P_p_I3I_PTorderingfinalstate_REG_[b[0]] ) continue;
      Manager()->FillHisto("48_DELTAR", _P_p_I1I_PTorderingfinalstate_REG_[a[0]]->dr(_P_p_I3I_PTorderingfinalstate_REG_[b[0]]));
    }
    }
  }
  }

  // Histogram number 49
  //   * Plot: DELTAR ( p[2] , p[3] ) 
  {
  {
    MAuint32 a[1];
    for (a[0]=0;a[0]<_P_p_I2I_PTorderingfinalstate_REG_.size();a[0]++)
    {
    MAuint32 b[1];
    for (b[0]=0;b[0]<_P_p_I3I_PTorderingfinalstate_REG_.size();b[0]++)
    {
     if ( _P_p_I2I_PTorderingfinalstate_REG_[a[0]] == _P_p_I3I_PTorderingfinalstate_REG_[b[0]] ) continue;
      Manager()->FillHisto("49_DELTAR", _P_p_I2I_PTorderingfinalstate_REG_[a[0]]->dr(_P_p_I3I_PTorderingfinalstate_REG_[b[0]]));
    }
    }
  }
  }

  return true;
}

void user::Finalize(const SampleFormat& summary, const std::vector<SampleFormat>& files)
{
}
