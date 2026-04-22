void selection_7()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo15","canvas_plotflow_tempo15",0,0,700,500);
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
  TH1F* S8_PT_0 = new TH1F("S8_PT_0","S8_PT_0",40,0.0,500.0);
  // Content
  S8_PT_0->SetBinContent(0,0.0); // underflow
  S8_PT_0->SetBinContent(1,0.0);
  S8_PT_0->SetBinContent(2,127291.797125281);
  S8_PT_0->SetBinContent(3,432011.69024358015);
  S8_PT_0->SetBinContent(4,469238.5894028592);
  S8_PT_0->SetBinContent(5,411296.7907113991);
  S8_PT_0->SetBinContent(6,308922.8930233799);
  S8_PT_0->SetBinContent(7,258786.69415564046);
  S8_PT_0->SetBinContent(8,199043.59550486034);
  S8_PT_0->SetBinContent(9,148306.99665068014);
  S8_PT_0->SetBinContent(10,130894.39704392097);
  S8_PT_0->SetBinContent(11,99071.47776260004);
  S8_PT_0->SetBinContent(12,69650.24842704009);
  S8_PT_0->SetBinContent(13,64846.78853551997);
  S8_PT_0->SetBinContent(14,48635.088901640025);
  S8_PT_0->SetBinContent(15,38727.93912538007);
  S8_PT_0->SetBinContent(16,29421.229335559958);
  S8_PT_0->SetBinContent(17,24317.5494508199);
  S8_PT_0->SetBinContent(18,22516.249491499908);
  S8_PT_0->SetBinContent(19,15911.47964066001);
  S8_PT_0->SetBinContent(20,14110.17968134002);
  S8_PT_0->SetBinContent(21,13809.959688120094);
  S8_PT_0->SetBinContent(22,9306.71478982);
  S8_PT_0->SetBinContent(23,7505.414830500007);
  S8_PT_0->SetBinContent(24,7805.6318237199985);
  S8_PT_0->SetBinContent(25,7805.6318237199985);
  S8_PT_0->SetBinContent(26,6904.981844060002);
  S8_PT_0->SetBinContent(27,4503.248898300004);
  S8_PT_0->SetBinContent(28,5403.898877960001);
  S8_PT_0->SetBinContent(29,3902.8159118599992);
  S8_PT_0->SetBinContent(30,3002.165932200003);
  S8_PT_0->SetBinContent(31,2101.5159525400068);
  S8_PT_0->SetBinContent(32,2101.5159525400068);
  S8_PT_0->SetBinContent(33,1801.2999593199927);
  S8_PT_0->SetBinContent(34,2101.5159525400068);
  S8_PT_0->SetBinContent(35,1200.8659728800103);
  S8_PT_0->SetBinContent(36,1801.2999593199927);
  S8_PT_0->SetBinContent(37,1501.0829661000016);
  S8_PT_0->SetBinContent(38,1200.8659728800103);
  S8_PT_0->SetBinContent(39,300.21659322000033);
  S8_PT_0->SetBinContent(40,600.4331864400007);
  S8_PT_0->SetBinContent(41,4503.248898300004); // overflow
  S8_PT_0->SetEntries(10000);
  // Style
  S8_PT_0->SetLineColor(9);
  S8_PT_0->SetLineStyle(1);
  S8_PT_0->SetLineWidth(1);
  S8_PT_0->SetFillColor(9);
  S8_PT_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_16","mystack");
  stack->Add(S8_PT_0);
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
  stack->GetXaxis()->SetTitle("p_{T} [ p_{1} ] (GeV/c) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_7.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_7.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_7.eps");

}
