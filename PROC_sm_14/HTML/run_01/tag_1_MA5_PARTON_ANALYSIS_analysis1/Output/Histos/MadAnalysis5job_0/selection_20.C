void selection_20()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo41","canvas_plotflow_tempo41",0,0,700,500);
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
  TH1F* S21_M_0 = new TH1F("S21_M_0","S21_M_0",40,0.0,500.0);
  // Content
  S21_M_0->SetBinContent(0,0.0); // underflow
  S21_M_0->SetBinContent(1,123389.01519055919);
  S21_M_0->SetBinContent(2,173525.22136288343);
  S21_M_0->SetBinContent(3,99071.49219680182);
  S21_M_0->SetBinContent(4,56740.94698544123);
  S21_M_0->SetBinContent(5,43831.63539616149);
  S21_M_0->SetBinContent(6,38727.94476784045);
  S21_M_0->SetBinContent(7,195741.2240979203);
  S21_M_0->SetBinContent(8,2129436.2621572716);
  S21_M_0->SetBinContent(9,79857.62983136182);
  S21_M_0->SetBinContent(10,20114.512476320047);
  S21_M_0->SetBinContent(11,13509.751663200585);
  S21_M_0->SetBinContent(12,6604.766813120204);
  S21_M_0->SetBinContent(13,5704.116702240165);
  S21_M_0->SetBinContent(14,3602.599443520033);
  S21_M_0->SetBinContent(15,2101.516258720009);
  S21_M_0->SetBinContent(16,2101.516258720009);
  S21_M_0->SetBinContent(17,1200.86614783997);
  S21_M_0->SetBinContent(18,1200.86614783997);
  S21_M_0->SetBinContent(19,600.4332739200096);
  S21_M_0->SetBinContent(20,1501.0831848000241);
  S21_M_0->SetBinContent(21,300.2166369600048);
  S21_M_0->SetBinContent(22,0.0);
  S21_M_0->SetBinContent(23,300.2166369600048);
  S21_M_0->SetBinContent(24,300.2166369600048);
  S21_M_0->SetBinContent(25,300.2166369600048);
  S21_M_0->SetBinContent(26,0.0);
  S21_M_0->SetBinContent(27,0.0);
  S21_M_0->SetBinContent(28,300.2166369600048);
  S21_M_0->SetBinContent(29,0.0);
  S21_M_0->SetBinContent(30,0.0);
  S21_M_0->SetBinContent(31,0.0);
  S21_M_0->SetBinContent(32,0.0);
  S21_M_0->SetBinContent(33,300.2166369600048);
  S21_M_0->SetBinContent(34,300.2166369600048);
  S21_M_0->SetBinContent(35,0.0);
  S21_M_0->SetBinContent(36,300.2166369600048);
  S21_M_0->SetBinContent(37,0.0);
  S21_M_0->SetBinContent(38,0.0);
  S21_M_0->SetBinContent(39,0.0);
  S21_M_0->SetBinContent(40,300.2166369600048);
  S21_M_0->SetBinContent(41,900.6499108800144); // overflow
  S21_M_0->SetEntries(10000);
  // Style
  S21_M_0->SetLineColor(9);
  S21_M_0->SetLineStyle(1);
  S21_M_0->SetLineWidth(1);
  S21_M_0->SetFillColor(9);
  S21_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_42","mystack");
  stack->Add(S21_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} l-_{1} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_20.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_20.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_20.eps");

}
