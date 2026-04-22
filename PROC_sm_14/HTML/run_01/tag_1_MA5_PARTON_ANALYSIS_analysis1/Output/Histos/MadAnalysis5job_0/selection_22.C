void selection_22()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo45","canvas_plotflow_tempo45",0,0,700,500);
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
  TH1F* S23_M_0 = new TH1F("S23_M_0","S23_M_0",40,0.0,500.0);
  // Content
  S23_M_0->SetBinContent(0,0.0); // underflow
  S23_M_0->SetBinContent(1,0.0);
  S23_M_0->SetBinContent(2,0.0);
  S23_M_0->SetBinContent(3,0.0);
  S23_M_0->SetBinContent(4,0.0);
  S23_M_0->SetBinContent(5,0.0);
  S23_M_0->SetBinContent(6,2401.732978399999);
  S23_M_0->SetBinContent(7,7205.1989351999955);
  S23_M_0->SetBinContent(8,9306.714916299998);
  S23_M_0->SetBinContent(9,15911.479856900001);
  S23_M_0->SetBinContent(10,16812.1298488);
  S23_M_0->SetBinContent(11,26719.27975969998);
  S23_M_0->SetBinContent(12,44732.27959769995);
  S23_M_0->SetBinContent(13,67548.73939249998);
  S23_M_0->SetBinContent(14,76555.23931149996);
  S23_M_0->SetBinContent(15,93667.57915760002);
  S23_M_0->SetBinContent(16,104475.39906039981);
  S23_M_0->SetBinContent(17,103874.8990658004);
  S23_M_0->SetBinContent(18,103574.69906850025);
  S23_M_0->SetBinContent(19,107177.29903610026);
  S23_M_0->SetBinContent(20,99371.69910629997);
  S23_M_0->SetBinContent(21,99671.91910359994);
  S23_M_0->SetBinContent(22,91566.06917649996);
  S23_M_0->SetBinContent(23,90665.41918459996);
  S23_M_0->SetBinContent(24,86462.37922240004);
  S23_M_0->SetBinContent(25,86162.16922509996);
  S23_M_0->SetBinContent(26,79557.3992845);
  S23_M_0->SetBinContent(27,80157.83927909994);
  S23_M_0->SetBinContent(28,73252.84934120002);
  S23_M_0->SetBinContent(29,64246.359422199945);
  S23_M_0->SetBinContent(30,61544.40944649995);
  S23_M_0->SetBinContent(31,64846.78941679998);
  S23_M_0->SetBinContent(32,56740.93948969999);
  S23_M_0->SetBinContent(33,54639.41950860002);
  S23_M_0->SetBinContent(34,45032.48959500001);
  S23_M_0->SetBinContent(35,49535.739554499996);
  S23_M_0->SetBinContent(36,48034.659567999974);
  S23_M_0->SetBinContent(37,53138.33952209999);
  S23_M_0->SetBinContent(38,45332.70959229998);
  S23_M_0->SetBinContent(39,42030.329621999954);
  S23_M_0->SetBinContent(40,41730.10962469999);
  S23_M_0->SetBinContent(41,808483.2927289002); // overflow
  S23_M_0->SetEntries(10000);
  // Style
  S23_M_0->SetLineColor(9);
  S23_M_0->SetLineStyle(1);
  S23_M_0->SetLineWidth(1);
  S23_M_0->SetFillColor(9);
  S23_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_46","mystack");
  stack->Add(S23_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} l-_{1} p_{1} p_{2} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_22.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_22.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_22.eps");

}
