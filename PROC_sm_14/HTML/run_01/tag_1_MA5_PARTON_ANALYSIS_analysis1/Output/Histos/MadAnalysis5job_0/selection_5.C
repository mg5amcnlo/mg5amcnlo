void selection_5()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo11","canvas_plotflow_tempo11",0,0,700,500);
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
  TH1F* S6_PT_0 = new TH1F("S6_PT_0","S6_PT_0",40,0.0,500.0);
  // Content
  S6_PT_0->SetBinContent(0,0.0); // underflow
  S6_PT_0->SetBinContent(1,113481.89062559874);
  S6_PT_0->SetBinContent(2,561104.853648798);
  S6_PT_0->SetBinContent(3,632256.1477712012);
  S6_PT_0->SetBinContent(4,546994.6548143994);
  S6_PT_0->SetBinContent(5,390881.96771040396);
  S6_PT_0->SetBinContent(6,245877.37968880142);
  S6_PT_0->SetBinContent(7,149507.88764959833);
  S6_PT_0->SetBinContent(8,113481.89062559874);
  S6_PT_0->SetBinContent(9,63045.48479200013);
  S6_PT_0->SetBinContent(10,41429.886577600366);
  S6_PT_0->SetBinContent(11,36626.426974399874);
  S6_PT_0->SetBinContent(12,28220.35766880024);
  S6_PT_0->SetBinContent(13,18613.42846240007);
  S6_PT_0->SetBinContent(14,11408.229057600149);
  S6_PT_0->SetBinContent(15,10207.359156800438);
  S6_PT_0->SetBinContent(16,9907.147181600054);
  S6_PT_0->SetBinContent(17,3902.815677600012);
  S6_PT_0->SetBinContent(18,6904.981429600034);
  S6_PT_0->SetBinContent(19,3602.598702400043);
  S6_PT_0->SetBinContent(20,2101.515826400032);
  S6_PT_0->SetBinContent(21,2401.732801600001);
  S6_PT_0->SetBinContent(22,1501.082876000011);
  S6_PT_0->SetBinContent(23,1200.8659008000418);
  S6_PT_0->SetBinContent(24,1501.082876000011);
  S6_PT_0->SetBinContent(25,300.21657520000224);
  S6_PT_0->SetBinContent(26,900.6497256000066);
  S6_PT_0->SetBinContent(27,600.4331504000045);
  S6_PT_0->SetBinContent(28,900.6497256000066);
  S6_PT_0->SetBinContent(29,600.4331504000045);
  S6_PT_0->SetBinContent(30,600.4331504000045);
  S6_PT_0->SetBinContent(31,900.6497256000066);
  S6_PT_0->SetBinContent(32,600.4331504000045);
  S6_PT_0->SetBinContent(33,0.0);
  S6_PT_0->SetBinContent(34,0.0);
  S6_PT_0->SetBinContent(35,0.0);
  S6_PT_0->SetBinContent(36,300.21657520000224);
  S6_PT_0->SetBinContent(37,0.0);
  S6_PT_0->SetBinContent(38,300.21657520000224);
  S6_PT_0->SetBinContent(39,0.0);
  S6_PT_0->SetBinContent(40,0.0);
  S6_PT_0->SetBinContent(41,0.0); // overflow
  S6_PT_0->SetEntries(10000);
  // Style
  S6_PT_0->SetLineColor(9);
  S6_PT_0->SetLineStyle(1);
  S6_PT_0->SetLineWidth(1);
  S6_PT_0->SetFillColor(9);
  S6_PT_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_12","mystack");
  stack->Add(S6_PT_0);
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
  stack->GetXaxis()->SetTitle("p_{T} [ l+_{1} ] (GeV/c) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_5.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_5.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_5.eps");

}
