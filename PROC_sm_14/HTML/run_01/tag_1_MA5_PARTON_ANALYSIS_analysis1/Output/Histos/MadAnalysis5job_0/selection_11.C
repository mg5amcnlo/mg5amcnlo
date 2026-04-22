void selection_11()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo23","canvas_plotflow_tempo23",0,0,700,500);
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
  TH1F* S12_PT_0 = new TH1F("S12_PT_0","S12_PT_0",40,0.0,500.0);
  // Content
  S12_PT_0->SetBinContent(0,0.0); // underflow
  S12_PT_0->SetBinContent(1,0.0);
  S12_PT_0->SetBinContent(2,2013252.6629564161);
  S12_PT_0->SetBinContent(3,623849.9955597366);
  S12_PT_0->SetBinContent(4,202345.96612476374);
  S12_PT_0->SetBinContent(5,79857.6066308415);
  S12_PT_0->SetBinContent(6,30922.304823220835);
  S12_PT_0->SetBinContent(7,17412.557084920958);
  S12_PT_0->SetBinContent(8,12609.097889079885);
  S12_PT_0->SetBinContent(9,7205.197793760101);
  S12_PT_0->SetBinContent(10,5103.681145580176);
  S12_PT_0->SetBinContent(11,3002.1654974000844);
  S12_PT_0->SetBinContent(12,1501.0827487000422);
  S12_PT_0->SetBinContent(13,1801.2996984399836);
  S12_PT_0->SetBinContent(14,900.6496492200253);
  S12_PT_0->SetBinContent(15,900.6496492200253);
  S12_PT_0->SetBinContent(16,300.2165497400085);
  S12_PT_0->SetBinContent(17,600.433099480017);
  S12_PT_0->SetBinContent(18,0.0);
  S12_PT_0->SetBinContent(19,0.0);
  S12_PT_0->SetBinContent(20,0.0);
  S12_PT_0->SetBinContent(21,0.0);
  S12_PT_0->SetBinContent(22,300.2165497400085);
  S12_PT_0->SetBinContent(23,0.0);
  S12_PT_0->SetBinContent(24,300.2165497400085);
  S12_PT_0->SetBinContent(25,0.0);
  S12_PT_0->SetBinContent(26,0.0);
  S12_PT_0->SetBinContent(27,0.0);
  S12_PT_0->SetBinContent(28,0.0);
  S12_PT_0->SetBinContent(29,0.0);
  S12_PT_0->SetBinContent(30,0.0);
  S12_PT_0->SetBinContent(31,0.0);
  S12_PT_0->SetBinContent(32,0.0);
  S12_PT_0->SetBinContent(33,0.0);
  S12_PT_0->SetBinContent(34,0.0);
  S12_PT_0->SetBinContent(35,0.0);
  S12_PT_0->SetBinContent(36,0.0);
  S12_PT_0->SetBinContent(37,0.0);
  S12_PT_0->SetBinContent(38,0.0);
  S12_PT_0->SetBinContent(39,0.0);
  S12_PT_0->SetBinContent(40,0.0);
  S12_PT_0->SetBinContent(41,0.0); // overflow
  S12_PT_0->SetEntries(10000);
  // Style
  S12_PT_0->SetLineColor(9);
  S12_PT_0->SetLineStyle(1);
  S12_PT_0->SetLineWidth(1);
  S12_PT_0->SetFillColor(9);
  S12_PT_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_24","mystack");
  stack->Add(S12_PT_0);
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
  stack->GetXaxis()->SetTitle("p_{T} [ p_{3} ] (GeV/c) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_11.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_11.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_11.eps");

}
