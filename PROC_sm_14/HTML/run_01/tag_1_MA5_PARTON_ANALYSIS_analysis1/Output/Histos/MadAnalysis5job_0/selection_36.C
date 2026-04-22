void selection_36()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo73","canvas_plotflow_tempo73",0,0,700,500);
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
  TH1F* S37_M_0 = new TH1F("S37_M_0","S37_M_0",40,0.0,500.0);
  // Content
  S37_M_0->SetBinContent(0,0.0); // underflow
  S37_M_0->SetBinContent(1,0.0);
  S37_M_0->SetBinContent(2,0.0);
  S37_M_0->SetBinContent(3,0.0);
  S37_M_0->SetBinContent(4,6604.76607480001);
  S37_M_0->SetBinContent(5,27619.93031280004);
  S37_M_0->SetBinContent(6,61544.4106970001);
  S37_M_0->SetBinContent(7,81358.70092140003);
  S37_M_0->SetBinContent(8,109278.80123759955);
  S37_M_0->SetBinContent(9,127892.30144840035);
  S37_M_0->SetBinContent(10,131795.10149260017);
  S37_M_0->SetBinContent(11,119786.40135659976);
  S37_M_0->SetBinContent(12,125790.80142460053);
  S37_M_0->SetBinContent(13,107177.30121379973);
  S37_M_0->SetBinContent(14,114682.70129879955);
  S37_M_0->SetBinContent(15,111080.10125799956);
  S37_M_0->SetBinContent(16,110479.70125119992);
  S37_M_0->SetBinContent(17,98771.27111860013);
  S37_M_0->SetBinContent(18,89164.33100980002);
  S37_M_0->SetBinContent(19,92766.93105060002);
  S37_M_0->SetBinContent(20,83460.22094520008);
  S37_M_0->SetBinContent(21,79557.40090100003);
  S37_M_0->SetBinContent(22,75654.59085680009);
  S37_M_0->SetBinContent(23,62445.06070720009);
  S37_M_0->SetBinContent(24,58242.020659600006);
  S37_M_0->SetBinContent(25,54939.64062220004);
  S37_M_0->SetBinContent(26,56140.510635800085);
  S37_M_0->SetBinContent(27,51637.26058480007);
  S37_M_0->SetBinContent(28,44432.060503200046);
  S37_M_0->SetBinContent(29,43231.190489600005);
  S37_M_0->SetBinContent(30,41730.11047260004);
  S37_M_0->SetBinContent(31,41429.8904692);
  S37_M_0->SetBinContent(32,33324.04037739998);
  S37_M_0->SetBinContent(33,36326.210411400025);
  S37_M_0->SetBinContent(34,38427.73043520007);
  S37_M_0->SetBinContent(35,31522.740356999977);
  S37_M_0->SetBinContent(36,24617.76027879999);
  S37_M_0->SetBinContent(37,32123.180363800053);
  S37_M_0->SetBinContent(38,24317.550275400066);
  S37_M_0->SetBinContent(39,28220.360319600004);
  S37_M_0->SetBinContent(40,23717.11026859999);
  S37_M_0->SetBinContent(41,520875.8058990001); // overflow
  S37_M_0->SetEntries(10000);
  // Style
  S37_M_0->SetLineColor(9);
  S37_M_0->SetLineStyle(1);
  S37_M_0->SetLineWidth(1);
  S37_M_0->SetFillColor(9);
  S37_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_74","mystack");
  stack->Add(S37_M_0);
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
  stack->GetXaxis()->SetTitle("M [ p_{1} p_{2} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_36.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_36.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_36.eps");

}
