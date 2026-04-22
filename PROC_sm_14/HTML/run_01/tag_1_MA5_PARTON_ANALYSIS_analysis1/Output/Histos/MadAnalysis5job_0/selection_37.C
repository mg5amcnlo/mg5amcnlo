void selection_37()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo75","canvas_plotflow_tempo75",0,0,700,500);
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
  TH1F* S38_M_0 = new TH1F("S38_M_0","S38_M_0",40,0.0,500.0);
  // Content
  S38_M_0->SetBinContent(0,0.0); // underflow
  S38_M_0->SetBinContent(1,0.0);
  S38_M_0->SetBinContent(2,49535.73950500001);
  S38_M_0->SetBinContent(3,169922.59830200006);
  S38_M_0->SetBinContent(4,227864.3977230001);
  S38_M_0->SetBinContent(5,271695.99728500034);
  S38_M_0->SetBinContent(6,268093.3973210004);
  S38_M_0->SetBinContent(7,256084.7974409997);
  S38_M_0->SetBinContent(8,208350.2979180003);
  S38_M_0->SetBinContent(9,168721.6983140004);
  S38_M_0->SetBinContent(10,150108.29850000006);
  S38_M_0->SetBinContent(11,134797.2986529996);
  S38_M_0->SetBinContent(12,111380.39888699964);
  S38_M_0->SetBinContent(13,98471.049016);
  S38_M_0->SetBinContent(14,84360.86915699998);
  S38_M_0->SetBinContent(15,75954.79924100004);
  S38_M_0->SetBinContent(16,60943.96939100003);
  S38_M_0->SetBinContent(17,55239.859447999974);
  S38_M_0->SetBinContent(18,47134.00952899999);
  S38_M_0->SetBinContent(19,36326.209637);
  S38_M_0->SetBinContent(20,38727.93961300004);
  S38_M_0->SetBinContent(21,35425.559646);
  S38_M_0->SetBinContent(22,34224.68965800005);
  S38_M_0->SetBinContent(23,32123.17967899998);
  S38_M_0->SetBinContent(24,24617.759754000024);
  S38_M_0->SetBinContent(25,26719.279732999985);
  S38_M_0->SetBinContent(26,22516.24977499996);
  S38_M_0->SetBinContent(27,25218.19974799996);
  S38_M_0->SetBinContent(28,17412.559826000037);
  S38_M_0->SetBinContent(29,13209.52986800001);
  S38_M_0->SetBinContent(30,18012.999819999968);
  S38_M_0->SetBinContent(31,13209.52986800001);
  S38_M_0->SetBinContent(32,12308.879877000012);
  S38_M_0->SetBinContent(33,12008.659880000047);
  S38_M_0->SetBinContent(34,10207.35989800005);
  S38_M_0->SetBinContent(35,9306.714907000001);
  S38_M_0->SetBinContent(36,7805.631922);
  S38_M_0->SetBinContent(37,10507.579895000015);
  S38_M_0->SetBinContent(38,9006.497910000004);
  S38_M_0->SetBinContent(39,6604.765933999995);
  S38_M_0->SetBinContent(40,8105.848918999996);
  S38_M_0->SetBinContent(41,139900.8986020004); // overflow
  S38_M_0->SetEntries(10000);
  // Style
  S38_M_0->SetLineColor(9);
  S38_M_0->SetLineStyle(1);
  S38_M_0->SetLineWidth(1);
  S38_M_0->SetFillColor(9);
  S38_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_76","mystack");
  stack->Add(S38_M_0);
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
  stack->GetXaxis()->SetTitle("M [ p_{1} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_37.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_37.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_37.eps");

}
