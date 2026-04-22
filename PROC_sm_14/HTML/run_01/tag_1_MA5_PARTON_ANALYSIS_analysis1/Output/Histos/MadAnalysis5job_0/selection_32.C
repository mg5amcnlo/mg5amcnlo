void selection_32()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo65","canvas_plotflow_tempo65",0,0,700,500);
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
  TH1F* S33_M_0 = new TH1F("S33_M_0","S33_M_0",40,0.0,500.0);
  // Content
  S33_M_0->SetBinContent(0,0.0); // underflow
  S33_M_0->SetBinContent(1,51036.81522300063);
  S33_M_0->SetBinContent(2,126090.98819799846);
  S33_M_0->SetBinContent(3,207149.48061099747);
  S33_M_0->SetBinContent(4,246778.076901797);
  S33_M_0->SetBinContent(5,270495.17468189826);
  S33_M_0->SetBinContent(6,267192.77499099984);
  S33_M_0->SetBinContent(7,242875.17726710485);
  S33_M_0->SetBinContent(8,207149.48061099747);
  S33_M_0->SetBinContent(9,172024.0838987026);
  S33_M_0->SetBinContent(10,149808.08597809976);
  S33_M_0->SetBinContent(11,129993.78783269998);
  S33_M_0->SetBinContent(12,102373.89041789719);
  S33_M_0->SetBinContent(13,88263.6717386008);
  S33_M_0->SetBinContent(14,74453.71303120034);
  S33_M_0->SetBinContent(15,67248.51370560043);
  S33_M_0->SetBinContent(16,53438.55499819997);
  S33_M_0->SetBinContent(17,52838.11505440061);
  S33_M_0->SetBinContent(18,48935.305419700024);
  S33_M_0->SetBinContent(19,34224.68679660052);
  S33_M_0->SetBinContent(20,35125.33671230051);
  S33_M_0->SetBinContent(21,36025.98662800049);
  S33_M_0->SetBinContent(22,28220.357358600282);
  S33_M_0->SetBinContent(23,21315.378004900052);
  S33_M_0->SetBinContent(24,20114.50811730038);
  S33_M_0->SetBinContent(25,18313.2082859004);
  S33_M_0->SetBinContent(26,18313.2082859004);
  S33_M_0->SetBinContent(27,22516.247892499727);
  S33_M_0->SetBinContent(28,21315.378004900052);
  S33_M_0->SetBinContent(29,10807.798988399867);
  S33_M_0->SetBinContent(30,10507.579016500184);
  S33_M_0->SetBinContent(31,9606.93110080001);
  S33_M_0->SetBinContent(32,9306.714128900043);
  S33_M_0->SetBinContent(33,9606.93110080001);
  S33_M_0->SetBinContent(34,7205.198325600006);
  S33_M_0->SetBinContent(35,7805.63126940003);
  S33_M_0->SetBinContent(36,8706.281185100019);
  S33_M_0->SetBinContent(37,6904.981353700041);
  S33_M_0->SetBinContent(38,5704.115466099993);
  S33_M_0->SetBinContent(39,7205.198325600006);
  S33_M_0->SetBinContent(40,5103.681522300062);
  S33_M_0->SetBinContent(41,90064.97157000078); // overflow
  S33_M_0->SetEntries(10000);
  // Style
  S33_M_0->SetLineColor(9);
  S33_M_0->SetLineStyle(1);
  S33_M_0->SetLineWidth(1);
  S33_M_0->SetFillColor(9);
  S33_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_66","mystack");
  stack->Add(S33_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l-_{1} p_{2} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_32.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_32.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_32.eps");

}
