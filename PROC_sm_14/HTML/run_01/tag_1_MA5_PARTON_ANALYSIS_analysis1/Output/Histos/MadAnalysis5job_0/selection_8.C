void selection_8()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo17","canvas_plotflow_tempo17",0,0,700,500);
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
  TH1F* S9_ETA_0 = new TH1F("S9_ETA_0","S9_ETA_0",40,-10.0,10.0);
  // Content
  S9_ETA_0->SetBinContent(0,0.0); // underflow
  S9_ETA_0->SetBinContent(1,0.0);
  S9_ETA_0->SetBinContent(2,0.0);
  S9_ETA_0->SetBinContent(3,0.0);
  S9_ETA_0->SetBinContent(4,0.0);
  S9_ETA_0->SetBinContent(5,0.0);
  S9_ETA_0->SetBinContent(6,0.0);
  S9_ETA_0->SetBinContent(7,0.0);
  S9_ETA_0->SetBinContent(8,0.0);
  S9_ETA_0->SetBinContent(9,0.0);
  S9_ETA_0->SetBinContent(10,0.0);
  S9_ETA_0->SetBinContent(11,4503.2490375);
  S9_ETA_0->SetBinContent(12,15611.260129999975);
  S9_ETA_0->SetBinContent(13,38127.51031750002);
  S9_ETA_0->SetBinContent(14,65447.22054500001);
  S9_ETA_0->SetBinContent(15,108078.0009000002);
  S9_ETA_0->SetBinContent(16,171723.90143000006);
  S9_ETA_0->SetBinContent(17,208350.30173499984);
  S9_ETA_0->SetBinContent(18,266292.1022174998);
  S9_ETA_0->SetBinContent(19,307121.6025575001);
  S9_ETA_0->SetBinContent(20,320631.3026699998);
  S9_ETA_0->SetBinContent(21,335342.0027925005);
  S9_ETA_0->SetBinContent(22,306220.90254999977);
  S9_ETA_0->SetBinContent(23,275298.6022924998);
  S9_ETA_0->SetBinContent(24,207749.9017300001);
  S9_ETA_0->SetBinContent(25,148006.80123250012);
  S9_ETA_0->SetBinContent(26,107477.50089499966);
  S9_ETA_0->SetBinContent(27,65747.44054750004);
  S9_ETA_0->SetBinContent(28,29421.230245000028);
  S9_ETA_0->SetBinContent(29,17112.350142500032);
  S9_ETA_0->SetBinContent(30,3902.8160325000017);
  S9_ETA_0->SetBinContent(31,0.0);
  S9_ETA_0->SetBinContent(32,0.0);
  S9_ETA_0->SetBinContent(33,0.0);
  S9_ETA_0->SetBinContent(34,0.0);
  S9_ETA_0->SetBinContent(35,0.0);
  S9_ETA_0->SetBinContent(36,0.0);
  S9_ETA_0->SetBinContent(37,0.0);
  S9_ETA_0->SetBinContent(38,0.0);
  S9_ETA_0->SetBinContent(39,0.0);
  S9_ETA_0->SetBinContent(40,0.0);
  S9_ETA_0->SetBinContent(41,0.0); // overflow
  S9_ETA_0->SetEntries(10000);
  // Style
  S9_ETA_0->SetLineColor(9);
  S9_ETA_0->SetLineStyle(1);
  S9_ETA_0->SetLineWidth(1);
  S9_ETA_0->SetFillColor(9);
  S9_ETA_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_18","mystack");
  stack->Add(S9_ETA_0);
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
  stack->GetXaxis()->SetTitle("#eta [ p_{1} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_8.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_8.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_8.eps");

}
