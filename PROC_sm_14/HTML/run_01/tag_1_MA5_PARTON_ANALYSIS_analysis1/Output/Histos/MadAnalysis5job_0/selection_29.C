void selection_29()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo59","canvas_plotflow_tempo59",0,0,700,500);
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
  TH1F* S30_M_0 = new TH1F("S30_M_0","S30_M_0",40,0.0,500.0);
  // Content
  S30_M_0->SetBinContent(0,0.0); // underflow
  S30_M_0->SetBinContent(1,0.0);
  S30_M_0->SetBinContent(2,0.0);
  S30_M_0->SetBinContent(3,1200.8659624000138);
  S30_M_0->SetBinContent(4,9006.49771800001);
  S30_M_0->SetBinContent(5,20414.729360799985);
  S30_M_0->SetBinContent(6,34825.128909599895);
  S30_M_0->SetBinContent(7,60643.758101199855);
  S30_M_0->SetBinContent(8,81058.48746199984);
  S30_M_0->SetBinContent(9,96669.74697319996);
  S30_M_0->SetBinContent(10,113782.09643739986);
  S30_M_0->SetBinContent(11,105075.79671000043);
  S30_M_0->SetBinContent(12,128792.89596740081);
  S30_M_0->SetBinContent(13,125190.29608020083);
  S30_M_0->SetBinContent(14,114082.29642800038);
  S30_M_0->SetBinContent(15,105976.49668179886);
  S30_M_0->SetBinContent(16,108678.3965972004);
  S30_M_0->SetBinContent(17,106276.69667239937);
  S30_M_0->SetBinContent(18,98771.26690739984);
  S30_M_0->SetBinContent(19,88864.1172175999);
  S30_M_0->SetBinContent(20,96369.52698260006);
  S30_M_0->SetBinContent(21,84360.86735859992);
  S30_M_0->SetBinContent(22,81358.69745260004);
  S30_M_0->SetBinContent(23,80458.04748080006);
  S30_M_0->SetBinContent(24,69950.46780980001);
  S30_M_0->SetBinContent(25,58242.01817640007);
  S30_M_0->SetBinContent(26,56140.50824219988);
  S30_M_0->SetBinContent(27,56140.50824219988);
  S30_M_0->SetBinContent(28,48334.868486600135);
  S30_M_0->SetBinContent(29,49835.95843959992);
  S30_M_0->SetBinContent(30,44131.838618200054);
  S30_M_0->SetBinContent(31,50736.60841139991);
  S30_M_0->SetBinContent(32,46833.78853360004);
  S30_M_0->SetBinContent(33,41730.10869339996);
  S30_M_0->SetBinContent(34,35425.5588908);
  S30_M_0->SetBinContent(35,36626.42885319989);
  S30_M_0->SetBinContent(36,30321.879050599928);
  S30_M_0->SetBinContent(37,31822.959003600023);
  S30_M_0->SetBinContent(38,33624.25894720001);
  S30_M_0->SetBinContent(39,29721.439069400138);
  S30_M_0->SetBinContent(40,27319.70914460005);
  S30_M_0->SetBinContent(41,513370.38392600016); // overflow
  S30_M_0->SetEntries(10000);
  // Style
  S30_M_0->SetLineColor(9);
  S30_M_0->SetLineStyle(1);
  S30_M_0->SetLineWidth(1);
  S30_M_0->SetFillColor(9);
  S30_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_60","mystack");
  stack->Add(S30_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l-_{1} p_{1} p_{2} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_29.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_29.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_29.eps");

}
