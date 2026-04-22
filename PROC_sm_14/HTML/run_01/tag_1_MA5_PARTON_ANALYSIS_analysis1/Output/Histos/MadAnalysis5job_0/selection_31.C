void selection_31()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo63","canvas_plotflow_tempo63",0,0,700,500);
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
  TH1F* S32_M_0 = new TH1F("S32_M_0","S32_M_0",40,0.0,500.0);
  // Content
  S32_M_0->SetBinContent(0,0.0); // underflow
  S32_M_0->SetBinContent(1,0.0);
  S32_M_0->SetBinContent(2,0.0);
  S32_M_0->SetBinContent(3,1801.299904599984);
  S32_M_0->SetBinContent(4,8105.848570699982);
  S32_M_0->SetBinContent(5,28820.788473600278);
  S32_M_0->SetBinContent(6,55540.067058500215);
  S32_M_0->SetBinContent(7,81658.91567519998);
  S32_M_0->SetBinContent(8,109579.09419649816);
  S32_M_0->SetBinContent(9,120687.09360819895);
  S32_M_0->SetBinContent(10,144103.99236799873);
  S32_M_0->SetBinContent(11,146205.4922566996);
  S32_M_0->SetBinContent(12,135998.09279730145);
  S32_M_0->SetBinContent(13,139300.49262240055);
  S32_M_0->SetBinContent(14,123388.99346510156);
  S32_M_0->SetBinContent(15,117084.49379899897);
  S32_M_0->SetBinContent(16,111080.09411700256);
  S32_M_0->SetBinContent(17,105075.79443500085);
  S32_M_0->SetBinContent(18,102373.89457809822);
  S32_M_0->SetBinContent(19,85861.94545260012);
  S32_M_0->SetBinContent(20,89764.76524589992);
  S32_M_0->SetBinContent(21,84961.29550030014);
  S32_M_0->SetBinContent(22,84060.64554800013);
  S32_M_0->SetBinContent(23,68449.38637479993);
  S32_M_0->SetBinContent(24,71451.54621580026);
  S32_M_0->SetBinContent(25,60343.5368041);
  S32_M_0->SetBinContent(26,61544.40674049981);
  S32_M_0->SetBinContent(27,50736.607312899905);
  S32_M_0->SetBinContent(28,49535.73737650009);
  S32_M_0->SetBinContent(29,45332.70759909995);
  S32_M_0->SetBinContent(30,35725.77810789986);
  S32_M_0->SetBinContent(31,42330.53775810016);
  S32_M_0->SetBinContent(32,38727.93794890019);
  S32_M_0->SetBinContent(33,35425.55812380004);
  S32_M_0->SetBinContent(34,37827.2879966002);
  S32_M_0->SetBinContent(35,30922.308362300082);
  S32_M_0->SetBinContent(36,29421.228441799918);
  S32_M_0->SetBinContent(37,24017.328727999968);
  S32_M_0->SetBinContent(38,21615.59885519981);
  S32_M_0->SetBinContent(39,20414.728918799996);
  S32_M_0->SetBinContent(40,19814.298950599827);
  S32_M_0->SetBinContent(41,383076.3797116002); // overflow
  S32_M_0->SetEntries(10000);
  // Style
  S32_M_0->SetLineColor(9);
  S32_M_0->SetLineStyle(1);
  S32_M_0->SetLineWidth(1);
  S32_M_0->SetFillColor(9);
  S32_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_64","mystack");
  stack->Add(S32_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l-_{1} p_{1} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_31.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_31.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_31.eps");

}
