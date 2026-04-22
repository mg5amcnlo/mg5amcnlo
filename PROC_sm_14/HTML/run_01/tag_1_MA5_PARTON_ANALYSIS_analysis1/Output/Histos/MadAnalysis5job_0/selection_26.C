void selection_26()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo53","canvas_plotflow_tempo53",0,0,700,500);
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
  TH1F* S27_M_0 = new TH1F("S27_M_0","S27_M_0",40,0.0,500.0);
  // Content
  S27_M_0->SetBinContent(0,0.0); // underflow
  S27_M_0->SetBinContent(1,0.0);
  S27_M_0->SetBinContent(2,0.0);
  S27_M_0->SetBinContent(3,600.4331749600011);
  S27_M_0->SetBinContent(4,3302.382862279989);
  S27_M_0->SetBinContent(5,7205.198699519987);
  S27_M_0->SetBinContent(6,15611.25934896016);
  S27_M_0->SetBinContent(7,27619.928848159932);
  S27_M_0->SetBinContent(8,29121.008785560058);
  S27_M_0->SetBinContent(9,34524.90856020002);
  S27_M_0->SetBinContent(10,33624.258597760025);
  S27_M_0->SetBinContent(11,74753.93688251986);
  S27_M_0->SetBinContent(12,121587.69492940117);
  S27_M_0->SetBinContent(13,136898.79429087896);
  S27_M_0->SetBinContent(14,160315.6933143188);
  S27_M_0->SetBinContent(15,149507.89376503887);
  S27_M_0->SetBinContent(16,139900.89416568173);
  S27_M_0->SetBinContent(17,150408.49372748096);
  S27_M_0->SetBinContent(18,119185.9950295598);
  S27_M_0->SetBinContent(19,119185.9950295598);
  S27_M_0->SetBinContent(20,105976.49558043851);
  S27_M_0->SetBinContent(21,96369.5259810801);
  S27_M_0->SetBinContent(22,95168.66603115984);
  S27_M_0->SetBinContent(23,84060.64649440006);
  S27_M_0->SetBinContent(24,74453.71689504);
  S27_M_0->SetBinContent(25,67248.51719552005);
  S27_M_0->SetBinContent(26,68749.59713292019);
  S27_M_0->SetBinContent(27,59142.66753356011);
  S27_M_0->SetBinContent(28,51337.03785908003);
  S27_M_0->SetBinContent(29,53438.557771439875);
  S27_M_0->SetBinContent(30,51637.25784655989);
  S27_M_0->SetBinContent(31,46833.788046880065);
  S27_M_0->SetBinContent(32,43531.40818459995);
  S27_M_0->SetBinContent(33,40529.23830980011);
  S27_M_0->SetBinContent(34,42930.97820963982);
  S27_M_0->SetBinContent(35,34224.68857272016);
  S27_M_0->SetBinContent(36,30021.658748000053);
  S27_M_0->SetBinContent(37,38127.50840995999);
  S27_M_0->SetBinContent(38,26419.05889824008);
  S27_M_0->SetBinContent(39,27319.70886068007);
  S27_M_0->SetBinContent(40,25818.628923279946);
  S27_M_0->SetBinContent(41,515471.878503161); // overflow
  S27_M_0->SetEntries(10000);
  // Style
  S27_M_0->SetLineColor(9);
  S27_M_0->SetLineStyle(1);
  S27_M_0->SetLineWidth(1);
  S27_M_0->SetFillColor(9);
  S27_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_54","mystack");
  stack->Add(S27_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} l-_{1} p_{2} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_26.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_26.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_26.eps");

}
