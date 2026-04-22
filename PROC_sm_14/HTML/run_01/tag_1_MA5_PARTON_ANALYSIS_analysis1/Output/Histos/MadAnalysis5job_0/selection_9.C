void selection_9()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo19","canvas_plotflow_tempo19",0,0,700,500);
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
  TH1F* S10_PT_0 = new TH1F("S10_PT_0","S10_PT_0",40,0.0,500.0);
  // Content
  S10_PT_0->SetBinContent(0,0.0); // underflow
  S10_PT_0->SetBinContent(1,0.0);
  S10_PT_0->SetBinContent(2,708210.96999352);
  S10_PT_0->SetBinContent(3,889841.9622979221);
  S10_PT_0->SetBinContent(4,506765.578528642);
  S10_PT_0->SetBinContent(5,297514.6873944786);
  S10_PT_0->SetBinContent(6,192739.09183375863);
  S10_PT_0->SetBinContent(7,111980.79525543991);
  S10_PT_0->SetBinContent(8,69350.03706167992);
  S10_PT_0->SetBinContent(9,61844.61737968013);
  S10_PT_0->SetBinContent(10,36326.20846088002);
  S10_PT_0->SetBinContent(11,25518.4089188001);
  S10_PT_0->SetBinContent(12,19213.859185920148);
  S10_PT_0->SetBinContent(13,17112.349274959877);
  S10_PT_0->SetBinContent(14,12308.879478480054);
  S10_PT_0->SetBinContent(15,9907.147580240013);
  S10_PT_0->SetBinContent(16,7505.414682000017);
  S10_PT_0->SetBinContent(17,7805.631669280001);
  S10_PT_0->SetBinContent(18,6604.765720159981);
  S10_PT_0->SetBinContent(19,3302.3828600799907);
  S10_PT_0->SetBinContent(20,3302.3828600799907);
  S10_PT_0->SetBinContent(21,2101.5159109600136);
  S10_PT_0->SetBinContent(22,1801.2999236799872);
  S10_PT_0->SetBinContent(23,1200.8659491200196);
  S10_PT_0->SetBinContent(24,300.21658728000074);
  S10_PT_0->SetBinContent(25,1501.0829364000035);
  S10_PT_0->SetBinContent(26,1501.0829364000035);
  S10_PT_0->SetBinContent(27,1801.2999236799872);
  S10_PT_0->SetBinContent(28,600.4331745600015);
  S10_PT_0->SetBinContent(29,900.6497618400022);
  S10_PT_0->SetBinContent(30,300.21658728000074);
  S10_PT_0->SetBinContent(31,0.0);
  S10_PT_0->SetBinContent(32,300.21658728000074);
  S10_PT_0->SetBinContent(33,300.21658728000074);
  S10_PT_0->SetBinContent(34,600.4331745600015);
  S10_PT_0->SetBinContent(35,300.21658728000074);
  S10_PT_0->SetBinContent(36,0.0);
  S10_PT_0->SetBinContent(37,300.21658728000074);
  S10_PT_0->SetBinContent(38,0.0);
  S10_PT_0->SetBinContent(39,300.21658728000074);
  S10_PT_0->SetBinContent(40,0.0);
  S10_PT_0->SetBinContent(41,900.6497618400022); // overflow
  S10_PT_0->SetEntries(10000);
  // Style
  S10_PT_0->SetLineColor(9);
  S10_PT_0->SetLineStyle(1);
  S10_PT_0->SetLineWidth(1);
  S10_PT_0->SetFillColor(9);
  S10_PT_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_20","mystack");
  stack->Add(S10_PT_0);
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
  stack->GetXaxis()->SetTitle("p_{T} [ p_{2} ] (GeV/c) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_9.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_9.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_9.eps");

}
