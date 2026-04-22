void selection_42()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo85","canvas_plotflow_tempo85",0,0,700,500);
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
  TH1F* S43_DELTAR_0 = new TH1F("S43_DELTAR_0","S43_DELTAR_0",40,0.0,10.0);
  // Content
  S43_DELTAR_0->SetBinContent(0,0.0); // underflow
  S43_DELTAR_0->SetBinContent(1,0.0);
  S43_DELTAR_0->SetBinContent(2,79857.61712719995);
  S43_DELTAR_0->SetBinContent(3,147406.39469719844);
  S43_DELTAR_0->SetBinContent(4,132395.49523720093);
  S43_DELTAR_0->SetBinContent(5,148607.19465400084);
  S43_DELTAR_0->SetBinContent(6,166920.3939952013);
  S43_DELTAR_0->SetBinContent(7,198743.3928503999);
  S43_DELTAR_0->SetBinContent(8,239873.09137079903);
  S43_DELTAR_0->SetBinContent(9,288207.88963200175);
  S43_DELTAR_0->SetBinContent(10,305920.688994801);
  S43_DELTAR_0->SetBinContent(11,365964.08683479816);
  S43_DELTAR_0->SetBinContent(12,398987.88564679923);
  S43_DELTAR_0->SetBinContent(13,336842.9878824014);
  S43_DELTAR_0->SetBinContent(14,114982.99586359864);
  S43_DELTAR_0->SetBinContent(15,48635.08825040004);
  S43_DELTAR_0->SetBinContent(16,20714.949254799863);
  S43_DELTAR_0->SetBinContent(17,6004.331784000009);
  S43_DELTAR_0->SetBinContent(18,2101.51592440001);
  S43_DELTAR_0->SetBinContent(19,0.0);
  S43_DELTAR_0->SetBinContent(20,0.0);
  S43_DELTAR_0->SetBinContent(21,0.0);
  S43_DELTAR_0->SetBinContent(22,0.0);
  S43_DELTAR_0->SetBinContent(23,0.0);
  S43_DELTAR_0->SetBinContent(24,0.0);
  S43_DELTAR_0->SetBinContent(25,0.0);
  S43_DELTAR_0->SetBinContent(26,0.0);
  S43_DELTAR_0->SetBinContent(27,0.0);
  S43_DELTAR_0->SetBinContent(28,0.0);
  S43_DELTAR_0->SetBinContent(29,0.0);
  S43_DELTAR_0->SetBinContent(30,0.0);
  S43_DELTAR_0->SetBinContent(31,0.0);
  S43_DELTAR_0->SetBinContent(32,0.0);
  S43_DELTAR_0->SetBinContent(33,0.0);
  S43_DELTAR_0->SetBinContent(34,0.0);
  S43_DELTAR_0->SetBinContent(35,0.0);
  S43_DELTAR_0->SetBinContent(36,0.0);
  S43_DELTAR_0->SetBinContent(37,0.0);
  S43_DELTAR_0->SetBinContent(38,0.0);
  S43_DELTAR_0->SetBinContent(39,0.0);
  S43_DELTAR_0->SetBinContent(40,0.0);
  S43_DELTAR_0->SetBinContent(41,0.0); // overflow
  S43_DELTAR_0->SetEntries(10000);
  // Style
  S43_DELTAR_0->SetLineColor(9);
  S43_DELTAR_0->SetLineStyle(1);
  S43_DELTAR_0->SetLineWidth(1);
  S43_DELTAR_0->SetFillColor(9);
  S43_DELTAR_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_86","mystack");
  stack->Add(S43_DELTAR_0);
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
  stack->GetXaxis()->SetTitle("#DeltaR [ l-_{1}, l+_{1} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_42.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_42.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_42.eps");

}
