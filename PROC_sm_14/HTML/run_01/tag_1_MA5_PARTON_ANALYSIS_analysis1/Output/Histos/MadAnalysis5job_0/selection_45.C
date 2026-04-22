void selection_45()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo91","canvas_plotflow_tempo91",0,0,700,500);
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
  TH1F* S46_DELTAR_0 = new TH1F("S46_DELTAR_0","S46_DELTAR_0",40,0.0,10.0);
  // Content
  S46_DELTAR_0->SetBinContent(0,0.0); // underflow
  S46_DELTAR_0->SetBinContent(1,14410.398851839842);
  S46_DELTAR_0->SetBinContent(2,45933.13634024029);
  S46_DELTAR_0->SetBinContent(3,73853.28411567998);
  S46_DELTAR_0->SetBinContent(4,95168.66241736003);
  S46_DELTAR_0->SetBinContent(5,121887.89028848398);
  S46_DELTAR_0->SetBinContent(6,145905.2883748784);
  S46_DELTAR_0->SetBinContent(7,177427.98586328205);
  S46_DELTAR_0->SetBinContent(8,180430.18562407937);
  S46_DELTAR_0->SetBinContent(9,223361.1822035176);
  S46_DELTAR_0->SetBinContent(10,234469.18131847878);
  S46_DELTAR_0->SetBinContent(11,253983.27976367722);
  S46_DELTAR_0->SetBinContent(12,267192.7787111997);
  S46_DELTAR_0->SetBinContent(13,266292.0787829637);
  S46_DELTAR_0->SetBinContent(14,180129.98564799802);
  S46_DELTAR_0->SetBinContent(15,147106.08827920372);
  S46_DELTAR_0->SetBinContent(16,120386.89040807735);
  S46_DELTAR_0->SetBinContent(17,89464.54287184036);
  S46_DELTAR_0->SetBinContent(18,81658.91349376016);
  S46_DELTAR_0->SetBinContent(19,60643.75516815987);
  S46_DELTAR_0->SetBinContent(20,56740.93547912018);
  S46_DELTAR_0->SetBinContent(21,42630.75660336007);
  S46_DELTAR_0->SetBinContent(22,35125.33720136041);
  S46_DELTAR_0->SetBinContent(23,27319.70782328023);
  S46_DELTAR_0->SetBinContent(24,21315.378301680033);
  S46_DELTAR_0->SetBinContent(25,14410.398851839842);
  S46_DELTAR_0->SetBinContent(26,9306.714258480031);
  S46_DELTAR_0->SetBinContent(27,8105.848354159992);
  S46_DELTAR_0->SetBinContent(28,2401.7328086400007);
  S46_DELTAR_0->SetBinContent(29,3002.1657608000205);
  S46_DELTAR_0->SetBinContent(30,1501.0828804000103);
  S46_DELTAR_0->SetBinContent(31,600.4331521600042);
  S46_DELTAR_0->SetBinContent(32,0.0);
  S46_DELTAR_0->SetBinContent(33,0.0);
  S46_DELTAR_0->SetBinContent(34,0.0);
  S46_DELTAR_0->SetBinContent(35,0.0);
  S46_DELTAR_0->SetBinContent(36,0.0);
  S46_DELTAR_0->SetBinContent(37,0.0);
  S46_DELTAR_0->SetBinContent(38,0.0);
  S46_DELTAR_0->SetBinContent(39,0.0);
  S46_DELTAR_0->SetBinContent(40,0.0);
  S46_DELTAR_0->SetBinContent(41,0.0); // overflow
  S46_DELTAR_0->SetEntries(10000);
  // Style
  S46_DELTAR_0->SetLineColor(9);
  S46_DELTAR_0->SetLineStyle(1);
  S46_DELTAR_0->SetLineWidth(1);
  S46_DELTAR_0->SetFillColor(9);
  S46_DELTAR_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_92","mystack");
  stack->Add(S46_DELTAR_0);
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
  stack->GetXaxis()->SetTitle("#DeltaR [ l-_{1}, p_{3} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_45.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_45.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_45.eps");

}
