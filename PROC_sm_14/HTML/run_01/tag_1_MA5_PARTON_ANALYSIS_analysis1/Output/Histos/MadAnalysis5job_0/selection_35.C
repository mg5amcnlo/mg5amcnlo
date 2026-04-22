void selection_35()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo71","canvas_plotflow_tempo71",0,0,700,500);
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
  TH1F* S36_M_0 = new TH1F("S36_M_0","S36_M_0",40,0.0,500.0);
  // Content
  S36_M_0->SetBinContent(0,0.0); // underflow
  S36_M_0->SetBinContent(1,0.0);
  S36_M_0->SetBinContent(2,64246.35550599979);
  S36_M_0->SetBinContent(3,128492.69101200097);
  S36_M_0->SetBinContent(4,151609.38939499954);
  S36_M_0->SetBinContent(5,180129.9873999981);
  S36_M_0->SetBinContent(6,187635.38687499918);
  S36_M_0->SetBinContent(7,181030.58733700158);
  S36_M_0->SetBinContent(8,173224.98788299932);
  S36_M_0->SetBinContent(9,163918.28853399825);
  S36_M_0->SetBinContent(10,144103.98991999848);
  S36_M_0->SetBinContent(11,132695.69071800326);
  S36_M_0->SetBinContent(12,108678.39239800118);
  S36_M_0->SetBinContent(13,106576.89254500002);
  S36_M_0->SetBinContent(14,94268.01340599994);
  S36_M_0->SetBinContent(15,87963.463847);
  S36_M_0->SetBinContent(16,84961.29405700027);
  S36_M_0->SetBinContent(17,75654.5847079999);
  S36_M_0->SetBinContent(18,68149.16523300021);
  S36_M_0->SetBinContent(19,60643.755757999825);
  S36_M_0->SetBinContent(20,54939.63615700012);
  S36_M_0->SetBinContent(21,43531.406955000006);
  S36_M_0->SetBinContent(22,43231.186976000245);
  S36_M_0->SetBinContent(23,41129.6771229998);
  S36_M_0->SetBinContent(24,36926.63741700031);
  S36_M_0->SetBinContent(25,34224.68760600034);
  S36_M_0->SetBinContent(26,30922.30783700014);
  S36_M_0->SetBinContent(27,28820.787984000395);
  S36_M_0->SetBinContent(28,27319.708089000174);
  S36_M_0->SetBinContent(29,26419.058152000187);
  S36_M_0->SetBinContent(30,28520.578004999934);
  S36_M_0->SetBinContent(31,23717.108341000214);
  S36_M_0->SetBinContent(32,20714.94855099978);
  S36_M_0->SetBinContent(33,19213.858656000262);
  S36_M_0->SetBinContent(34,18012.99873999981);
  S36_M_0->SetBinContent(35,18313.20871900027);
  S36_M_0->SetBinContent(36,21315.37850900001);
  S36_M_0->SetBinContent(37,19514.078635000027);
  S36_M_0->SetBinContent(38,20114.50859300025);
  S36_M_0->SetBinContent(39,11408.229202000111);
  S36_M_0->SetBinContent(40,12909.309097000329);
  S36_M_0->SetBinContent(41,226963.78412399758); // overflow
  S36_M_0->SetEntries(10000);
  // Style
  S36_M_0->SetLineColor(9);
  S36_M_0->SetLineStyle(1);
  S36_M_0->SetLineWidth(1);
  S36_M_0->SetFillColor(9);
  S36_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_72","mystack");
  stack->Add(S36_M_0);
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
  stack->GetXaxis()->SetTitle("M [ p_{1} p_{2} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_35.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_35.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_35.eps");

}
