void selection_25()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo51","canvas_plotflow_tempo51",0,0,700,500);
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
  TH1F* S26_M_0 = new TH1F("S26_M_0","S26_M_0",40,0.0,500.0);
  // Content
  S26_M_0->SetBinContent(0,0.0); // underflow
  S26_M_0->SetBinContent(1,1801.2999579999914);
  S26_M_0->SetBinContent(2,18913.64955899991);
  S26_M_0->SetBinContent(3,30021.659300000014);
  S26_M_0->SetBinContent(4,38727.93909700005);
  S26_M_0->SetBinContent(5,51036.81881000007);
  S26_M_0->SetBinContent(6,48334.86887300009);
  S26_M_0->SetBinContent(7,51337.03880299999);
  S26_M_0->SetBinContent(8,49235.518852000074);
  S26_M_0->SetBinContent(9,116483.997284001);
  S26_M_0->SetBinContent(10,283404.49339199945);
  S26_M_0->SetBinContent(11,289108.59325899975);
  S26_M_0->SetBinContent(12,249179.79418999958);
  S26_M_0->SetBinContent(13,208350.29514200054);
  S26_M_0->SetBinContent(14,181030.59577900032);
  S26_M_0->SetBinContent(15,140801.59671699972);
  S26_M_0->SetBinContent(16,126991.59703900057);
  S26_M_0->SetBinContent(17,113481.89735399945);
  S26_M_0->SetBinContent(18,93367.36782299986);
  S26_M_0->SetBinContent(19,80758.26811699993);
  S26_M_0->SetBinContent(20,68449.38840399991);
  S26_M_0->SetBinContent(21,60943.96857900002);
  S26_M_0->SetBinContent(22,54339.2087329999);
  S26_M_0->SetBinContent(23,46233.358921999934);
  S26_M_0->SetBinContent(24,42630.75900599996);
  S26_M_0->SetBinContent(25,42630.75900599996);
  S26_M_0->SetBinContent(26,36626.4291459999);
  S26_M_0->SetBinContent(27,31222.529271999927);
  S26_M_0->SetBinContent(28,27619.929355999946);
  S26_M_0->SetBinContent(29,31522.739265000084);
  S26_M_0->SetBinContent(30,30622.08928600009);
  S26_M_0->SetBinContent(31,23717.109447000043);
  S26_M_0->SetBinContent(32,21015.159510000056);
  S26_M_0->SetBinContent(33,17112.349600999918);
  S26_M_0->SetBinContent(34,17412.559594000075);
  S26_M_0->SetBinContent(35,16511.909615000077);
  S26_M_0->SetBinContent(36,18313.20957300007);
  S26_M_0->SetBinContent(37,15010.829650000007);
  S26_M_0->SetBinContent(38,12909.309699000094);
  S26_M_0->SetBinContent(39,12909.309699000094);
  S26_M_0->SetBinContent(40,11108.009741000104);
  S26_M_0->SetBinContent(41,190937.7955479991); // overflow
  S26_M_0->SetEntries(10000);
  // Style
  S26_M_0->SetLineColor(9);
  S26_M_0->SetLineStyle(1);
  S26_M_0->SetLineWidth(1);
  S26_M_0->SetFillColor(9);
  S26_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_52","mystack");
  stack->Add(S26_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} l-_{1} p_{2} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_25.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_25.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_25.eps");

}
