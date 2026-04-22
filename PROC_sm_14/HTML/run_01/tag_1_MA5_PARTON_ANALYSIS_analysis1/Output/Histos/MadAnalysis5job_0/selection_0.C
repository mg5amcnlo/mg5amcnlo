void selection_0()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo1","canvas_plotflow_tempo1",0,0,700,500);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  canvas->SetHighLightColor(2);
  canvas->SetFillColor(0);
  canvas->SetBorderMode(0);
  canvas->SetBorderSize(3);
  canvas->SetFrameBorderMode(0);
  canvas->SetFrameBorderSize(0);
  canvas->SetTickx(1);
  canvas->SetTicky(1);
  canvas->SetLeftMargin(0.14);
  canvas->SetRightMargin(0.05);
  canvas->SetBottomMargin(0.15);
  canvas->SetTopMargin(0.05);

  // Creating a new TH1F
  TH1F* S1_THT_0 = new TH1F("S1_THT_0","S1_THT_0",40,0.0,500.0);
  // Content
  S1_THT_0->SetBinContent(0,0.0); // underflow
  S1_THT_0->SetBinContent(1,0.0);
  S1_THT_0->SetBinContent(2,0.0);
  S1_THT_0->SetBinContent(3,0.0);
  S1_THT_0->SetBinContent(4,4803.465894399995);
  S1_THT_0->SetBinContent(5,114982.99747219916);
  S1_THT_0->SetBinContent(6,241974.59468039972);
  S1_THT_0->SetBinContent(7,293912.09353859915);
  S1_THT_0->SetBinContent(8,297214.39346600097);
  S1_THT_0->SetBinContent(9,280702.4938290007);
  S1_THT_0->SetBinContent(10,237471.29477940084);
  S1_THT_0->SetBinContent(11,205348.19548559916);
  S1_THT_0->SetBinContent(12,164518.69638320006);
  S1_THT_0->SetBinContent(13,159715.1964888008);
  S1_THT_0->SetBinContent(14,133296.19706959947);
  S1_THT_0->SetBinContent(15,117985.09740620061);
  S1_THT_0->SetBinContent(16,96369.52788140003);
  S1_THT_0->SetBinContent(17,85561.72811900008);
  S1_THT_0->SetBinContent(18,71151.33843579992);
  S1_THT_0->SetBinContent(19,62745.26862060003);
  S1_THT_0->SetBinContent(20,55540.06877900006);
  S1_THT_0->SetBinContent(21,47734.43895060002);
  S1_THT_0->SetBinContent(22,36326.20920139999);
  S1_THT_0->SetBinContent(23,33324.03926740008);
  S1_THT_0->SetBinContent(24,28220.35937960003);
  S1_THT_0->SetBinContent(25,22516.249504999905);
  S1_THT_0->SetBinContent(26,18313.20959740007);
  S1_THT_0->SetBinContent(27,15010.82967000001);
  S1_THT_0->SetBinContent(28,18913.649584199924);
  S1_THT_0->SetBinContent(29,12909.309716200092);
  S1_THT_0->SetBinContent(30,14110.179689800016);
  S1_THT_0->SetBinContent(31,12308.879729400021);
  S1_THT_0->SetBinContent(32,13209.52970960002);
  S1_THT_0->SetBinContent(33,9606.93178879999);
  S1_THT_0->SetBinContent(34,7205.198841599991);
  S1_THT_0->SetBinContent(35,10807.799762399954);
  S1_THT_0->SetBinContent(36,8706.281808599992);
  S1_THT_0->SetBinContent(37,5704.115874599991);
  S1_THT_0->SetBinContent(38,4503.248901000003);
  S1_THT_0->SetBinContent(39,6304.548861399996);
  S1_THT_0->SetBinContent(40,3302.3829273999936);
  S1_THT_0->SetBinContent(41,49835.95890439994); // overflow
  S1_THT_0->SetEntries(10000);
  // Style
  S1_THT_0->SetLineColor(9);
  S1_THT_0->SetLineStyle(1);
  S1_THT_0->SetLineWidth(1);
  S1_THT_0->SetFillColor(9);
  S1_THT_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_2","mystack");
  stack->Add(S1_THT_0);
  stack->Draw("");

  // Y axis
  stack->GetYaxis()->SetLabelSize(0.04);
  stack->GetYaxis()->SetLabelOffset(0.005);
  stack->GetYaxis()->SetTitleSize(0.06);
  stack->GetYaxis()->SetTitleFont(22);
  stack->GetYaxis()->SetTitleOffset(1);
  stack->GetYaxis()->SetTitle("Events  ( L_{int} = 10 fb^{-1} )");

  // X axis
  stack->GetXaxis()->SetLabelSize(0.04);
  stack->GetXaxis()->SetLabelOffset(0.005);
  stack->GetXaxis()->SetTitleSize(0.06);
  stack->GetXaxis()->SetTitleFont(22);
  stack->GetXaxis()->SetTitleOffset(1);
  stack->GetXaxis()->SetTitle("H_{T} (GeV) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_0.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_0.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_0.eps");

}
