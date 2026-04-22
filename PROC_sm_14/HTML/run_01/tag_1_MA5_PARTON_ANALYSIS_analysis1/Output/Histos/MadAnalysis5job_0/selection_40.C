void selection_40()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo81","canvas_plotflow_tempo81",0,0,700,500);
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
  TH1F* S41_DELTAR_0 = new TH1F("S41_DELTAR_0","S41_DELTAR_0",40,0.0,10.0);
  // Content
  S41_DELTAR_0->SetBinContent(0,0.0); // underflow
  S41_DELTAR_0->SetBinContent(1,22816.459854080018);
  S41_DELTAR_0->SetBinContent(2,64846.78958527999);
  S41_DELTAR_0->SetBinContent(3,80458.04948544002);
  S41_DELTAR_0->SetBinContent(4,115883.59925888009);
  S41_DELTAR_0->SetBinContent(5,151909.59902848006);
  S41_DELTAR_0->SetBinContent(6,170823.2989075197);
  S41_DELTAR_0->SetBinContent(7,191538.19877504001);
  S41_DELTAR_0->SetBinContent(8,197842.69873472032);
  S41_DELTAR_0->SetBinContent(9,230266.09852736027);
  S41_DELTAR_0->SetBinContent(10,244076.09843904004);
  S41_DELTAR_0->SetBinContent(11,265091.2983046398);
  S41_DELTAR_0->SetBinContent(12,275298.5982393602);
  S41_DELTAR_0->SetBinContent(13,255484.29836608024);
  S41_DELTAR_0->SetBinContent(14,175026.2988806399);
  S41_DELTAR_0->SetBinContent(15,125790.79919551975);
  S41_DELTAR_0->SetBinContent(16,100272.29935872032);
  S41_DELTAR_0->SetBinContent(17,78056.3195008);
  S41_DELTAR_0->SetBinContent(18,55540.06964480002);
  S41_DELTAR_0->SetBinContent(19,52237.68966592001);
  S41_DELTAR_0->SetBinContent(20,43231.189723520016);
  S41_DELTAR_0->SetBinContent(21,27619.92982335999);
  S41_DELTAR_0->SetBinContent(22,26419.059831040016);
  S41_DELTAR_0->SetBinContent(23,20714.94986751998);
  S41_DELTAR_0->SetBinContent(24,11408.229927040009);
  S41_DELTAR_0->SetBinContent(25,6004.331961600003);
  S41_DELTAR_0->SetBinContent(26,5704.115963519998);
  S41_DELTAR_0->SetBinContent(27,3602.5989769600023);
  S41_DELTAR_0->SetBinContent(28,2401.73298464);
  S41_DELTAR_0->SetBinContent(29,1200.865992320003);
  S41_DELTAR_0->SetBinContent(30,300.2165980800001);
  S41_DELTAR_0->SetBinContent(31,300.2165980800001);
  S41_DELTAR_0->SetBinContent(32,0.0);
  S41_DELTAR_0->SetBinContent(33,0.0);
  S41_DELTAR_0->SetBinContent(34,0.0);
  S41_DELTAR_0->SetBinContent(35,0.0);
  S41_DELTAR_0->SetBinContent(36,0.0);
  S41_DELTAR_0->SetBinContent(37,0.0);
  S41_DELTAR_0->SetBinContent(38,0.0);
  S41_DELTAR_0->SetBinContent(39,0.0);
  S41_DELTAR_0->SetBinContent(40,0.0);
  S41_DELTAR_0->SetBinContent(41,0.0); // overflow
  S41_DELTAR_0->SetEntries(10000);
  // Style
  S41_DELTAR_0->SetLineColor(9);
  S41_DELTAR_0->SetLineStyle(1);
  S41_DELTAR_0->SetLineWidth(1);
  S41_DELTAR_0->SetFillColor(9);
  S41_DELTAR_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_82","mystack");
  stack->Add(S41_DELTAR_0);
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
  stack->GetXaxis()->SetTitle("#DeltaR [ l+_{1}, p_{2} ] ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_40.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_40.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_40.eps");

}
