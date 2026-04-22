void selection_21()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo43","canvas_plotflow_tempo43",0,0,700,500);
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
  TH1F* S22_M_0 = new TH1F("S22_M_0","S22_M_0",40,0.0,500.0);
  // Content
  S22_M_0->SetBinContent(0,0.0); // underflow
  S22_M_0->SetBinContent(1,0.0);
  S22_M_0->SetBinContent(2,1200.8659332000261);
  S22_M_0->SetBinContent(3,3902.815782900001);
  S22_M_0->SetBinContent(4,8406.064532400016);
  S22_M_0->SetBinContent(5,26419.05853040013);
  S22_M_0->SetBinContent(6,36025.98799600022);
  S22_M_0->SetBinContent(7,38127.50787910002);
  S22_M_0->SetBinContent(8,39628.587795600186);
  S22_M_0->SetBinContent(9,54939.63694390005);
  S22_M_0->SetBinContent(10,110779.89383770176);
  S22_M_0->SetBinContent(11,154011.09143290136);
  S22_M_0->SetBinContent(12,191237.98936209918);
  S22_M_0->SetBinContent(13,176227.19019709746);
  S22_M_0->SetBinContent(14,180129.98997999835);
  S22_M_0->SetBinContent(15,164218.49086509942);
  S22_M_0->SetBinContent(16,163618.09089849758);
  S22_M_0->SetBinContent(17,142302.6920841987);
  S22_M_0->SetBinContent(18,126090.99298599883);
  S22_M_0->SetBinContent(19,101473.19435540092);
  S22_M_0->SetBinContent(20,105676.19412160273);
  S22_M_0->SetBinContent(21,89464.5450234001);
  S22_M_0->SetBinContent(22,91866.27488980026);
  S22_M_0->SetBinContent(23,71751.76600870008);
  S22_M_0->SetBinContent(24,67548.73624249994);
  S22_M_0->SetBinContent(25,62745.26650970016);
  S22_M_0->SetBinContent(26,59142.666710100195);
  S22_M_0->SetBinContent(27,57941.80677689984);
  S22_M_0->SetBinContent(28,51036.81716100027);
  S22_M_0->SetBinContent(29,45632.927461599764);
  S22_M_0->SetBinContent(30,45933.13744490013);
  S22_M_0->SetBinContent(31,37226.85792920003);
  S22_M_0->SetBinContent(32,32723.60817970007);
  S22_M_0->SetBinContent(33,27920.148446899744);
  S22_M_0->SetBinContent(34,29721.43834670028);
  S22_M_0->SetBinContent(35,30321.87831329991);
  S22_M_0->SetBinContent(36,21015.15883100018);
  S22_M_0->SetBinContent(37,25818.62856379995);
  S22_M_0->SetBinContent(38,19514.07891450001);
  S22_M_0->SetBinContent(39,17712.77901470002);
  S22_M_0->SetBinContent(40,17412.55903140021);
  S22_M_0->SetBinContent(41,275298.58468610205); // overflow
  S22_M_0->SetEntries(10000);
  // Style
  S22_M_0->SetLineColor(9);
  S22_M_0->SetLineStyle(1);
  S22_M_0->SetLineWidth(1);
  S22_M_0->SetFillColor(9);
  S22_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_44","mystack");
  stack->Add(S22_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} l-_{1} p_{1} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_21.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_21.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_21.eps");

}
