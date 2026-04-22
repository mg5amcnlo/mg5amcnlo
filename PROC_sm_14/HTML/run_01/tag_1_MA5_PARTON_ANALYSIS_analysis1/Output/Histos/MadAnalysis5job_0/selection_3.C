void selection_3()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo7","canvas_plotflow_tempo7",0,0,700,500);
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
  TH1F* S4_PT_0 = new TH1F("S4_PT_0","S4_PT_0",40,0.0,500.0);
  // Content
  S4_PT_0->SetBinContent(0,0.0); // underflow
  S4_PT_0->SetBinContent(1,126691.39875088014);
  S4_PT_0->SetBinContent(2,538288.39469272);
  S4_PT_0->SetBinContent(3,618145.9939053602);
  S4_PT_0->SetBinContent(4,546394.1946128005);
  S4_PT_0->SetBinContent(5,397486.79608096);
  S4_PT_0->SetBinContent(6,255484.2974810404);
  S4_PT_0->SetBinContent(7,152209.79849928024);
  S4_PT_0->SetBinContent(8,99071.47902320004);
  S4_PT_0->SetBinContent(9,67548.739334);
  S4_PT_0->SetBinContent(10,51937.469487920054);
  S4_PT_0->SetBinContent(11,42930.97957671997);
  S4_PT_0->SetBinContent(12,25818.629745439994);
  S4_PT_0->SetBinContent(13,15010.82985200001);
  S4_PT_0->SetBinContent(14,13809.959863840044);
  S4_PT_0->SetBinContent(15,13209.529869760014);
  S4_PT_0->SetBinContent(16,7205.198928959999);
  S4_PT_0->SetBinContent(17,6604.765934879996);
  S4_PT_0->SetBinContent(18,4803.46595264);
  S4_PT_0->SetBinContent(19,4803.46595264);
  S4_PT_0->SetBinContent(20,2101.5159792800036);
  S4_PT_0->SetBinContent(21,3602.5989644800047);
  S4_PT_0->SetBinContent(22,1501.082985200001);
  S4_PT_0->SetBinContent(23,300.21659704000024);
  S4_PT_0->SetBinContent(24,600.4331940800005);
  S4_PT_0->SetBinContent(25,600.4331940800005);
  S4_PT_0->SetBinContent(26,900.6497911200006);
  S4_PT_0->SetBinContent(27,0.0);
  S4_PT_0->SetBinContent(28,1200.8659881600047);
  S4_PT_0->SetBinContent(29,1501.082985200001);
  S4_PT_0->SetBinContent(30,300.21659704000024);
  S4_PT_0->SetBinContent(31,600.4331940800005);
  S4_PT_0->SetBinContent(32,0.0);
  S4_PT_0->SetBinContent(33,0.0);
  S4_PT_0->SetBinContent(34,300.21659704000024);
  S4_PT_0->SetBinContent(35,0.0);
  S4_PT_0->SetBinContent(36,300.21659704000024);
  S4_PT_0->SetBinContent(37,0.0);
  S4_PT_0->SetBinContent(38,300.21659704000024);
  S4_PT_0->SetBinContent(39,0.0);
  S4_PT_0->SetBinContent(40,0.0);
  S4_PT_0->SetBinContent(41,600.4331940800005); // overflow
  S4_PT_0->SetEntries(10000);
  // Style
  S4_PT_0->SetLineColor(9);
  S4_PT_0->SetLineStyle(1);
  S4_PT_0->SetLineWidth(1);
  S4_PT_0->SetFillColor(9);
  S4_PT_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_8","mystack");
  stack->Add(S4_PT_0);
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
  stack->GetXaxis()->SetTitle("p_{T} [ l-_{1} ] (GeV/c) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_3.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_3.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_3.eps");

}
