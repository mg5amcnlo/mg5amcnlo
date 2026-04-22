void selection_48()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo97","canvas_plotflow_tempo97",0,0,700,500);
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
  TH1F* S49_DELTAR_0 = new TH1F("S49_DELTAR_0","S49_DELTAR_0",40,0.0,10.0);
  // Content
  S49_DELTAR_0->SetBinContent(0,0.0); // underflow
  S49_DELTAR_0->SetBinContent(1,3002.165728200026);
  S49_DELTAR_0->SetBinContent(2,24617.757771240318);
  S49_DELTAR_0->SetBinContent(3,71151.33355834009);
  S49_DELTAR_0->SetBinContent(4,112581.18980750324);
  S49_DELTAR_0->SetBinContent(5,151008.98632845675);
  S49_DELTAR_0->SetBinContent(6,162116.98532279814);
  S49_DELTAR_0->SetBinContent(7,161816.78534997662);
  S49_DELTAR_0->SetBinContent(8,189736.88282224085);
  S49_DELTAR_0->SetBinContent(9,191237.98268633932);
  S49_DELTAR_0->SetBinContent(10,214354.68059347756);
  S49_DELTAR_0->SetBinContent(11,230866.57909857886);
  S49_DELTAR_0->SetBinContent(12,237471.27850062482);
  S49_DELTAR_0->SetBinContent(13,242274.77806574173);
  S49_DELTAR_0->SetBinContent(14,173825.38426278252);
  S49_DELTAR_0->SetBinContent(15,154611.58600229674);
  S49_DELTAR_0->SetBinContent(16,133596.38790489998);
  S49_DELTAR_0->SetBinContent(17,102974.29067726033);
  S49_DELTAR_0->SetBinContent(18,78956.9628516603);
  S49_DELTAR_0->SetBinContent(19,76255.01309628034);
  S49_DELTAR_0->SetBinContent(20,56740.934862980255);
  S49_DELTAR_0->SetBinContent(21,56140.50491733996);
  S49_DELTAR_0->SetBinContent(22,43231.186086080415);
  S49_DELTAR_0->SetBinContent(23,32423.387064560535);
  S49_DELTAR_0->SetBinContent(24,21915.808015860355);
  S49_DELTAR_0->SetBinContent(25,19213.858260480385);
  S49_DELTAR_0->SetBinContent(26,15311.048613819823);
  S49_DELTAR_0->SetBinContent(27,11108.008994340476);
  S49_DELTAR_0->SetBinContent(28,11108.008994340476);
  S49_DELTAR_0->SetBinContent(29,7505.414320500065);
  S49_DELTAR_0->SetBinContent(30,4203.032619479983);
  S49_DELTAR_0->SetBinContent(31,3302.382701019992);
  S49_DELTAR_0->SetBinContent(32,1501.082864100013);
  S49_DELTAR_0->SetBinContent(33,2401.7327825600028);
  S49_DELTAR_0->SetBinContent(34,900.6497184600078);
  S49_DELTAR_0->SetBinContent(35,1200.8658912800465);
  S49_DELTAR_0->SetBinContent(36,900.6497184600078);
  S49_DELTAR_0->SetBinContent(37,300.2165728200026);
  S49_DELTAR_0->SetBinContent(38,300.2165728200026);
  S49_DELTAR_0->SetBinContent(39,0.0);
  S49_DELTAR_0->SetBinContent(40,0.0);
  S49_DELTAR_0->SetBinContent(41,0.0); // overflow
  S49_DELTAR_0->SetEntries(10000);
  // Style
  S49_DELTAR_0->SetLineColor(9);
  S49_DELTAR_0->SetLineStyle(1);
  S49_DELTAR_0->SetLineWidth(1);
  S49_DELTAR_0->SetFillColor(9);
  S49_DELTAR_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_98","mystack");
  stack->Add(S49_DELTAR_0);
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
  stack->GetXaxis()->SetTitle("#DeltaR [ p_{2}, p_{3} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_48.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_48.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_48.eps");

}
