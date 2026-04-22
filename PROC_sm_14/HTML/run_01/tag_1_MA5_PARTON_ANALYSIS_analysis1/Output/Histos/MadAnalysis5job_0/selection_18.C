void selection_18()
{

  // ROOT version
  Int_t root_version = gROOT->GetVersionInt();

  // Creating a new TCanvas
  TCanvas* canvas = new TCanvas("canvas_plotflow_tempo37","canvas_plotflow_tempo37",0,0,700,500);
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
  TH1F* S19_M_0 = new TH1F("S19_M_0","S19_M_0",40,0.0,500.0);
  // Content
  S19_M_0->SetBinContent(0,0.0); // underflow
  S19_M_0->SetBinContent(1,0.0);
  S19_M_0->SetBinContent(2,300.2165783400015);
  S19_M_0->SetBinContent(3,11708.44915525987);
  S19_M_0->SetBinContent(4,28520.577942299926);
  S19_M_0->SetBinContent(5,69350.03499653994);
  S19_M_0->SetBinContent(6,109278.7921157636);
  S19_M_0->SetBinContent(7,152209.7890183819);
  S19_M_0->SetBinContent(8,173525.1874805205);
  S19_M_0->SetBinContent(9,170823.28767545687);
  S19_M_0->SetBinContent(10,175626.68732890167);
  S19_M_0->SetBinContent(11,162116.9883035982);
  S19_M_0->SetBinContent(12,154911.78882343826);
  S19_M_0->SetBinContent(13,140201.18988477724);
  S19_M_0->SetBinContent(14,118885.79142263868);
  S19_M_0->SetBinContent(15,117684.8915092811);
  S19_M_0->SetBinContent(16,106576.89231070002);
  S19_M_0->SetBinContent(17,99371.6928305401);
  S19_M_0->SetBinContent(18,90965.62343702043);
  S19_M_0->SetBinContent(19,79557.39426010031);
  S19_M_0->SetBinContent(20,71451.54484492041);
  S19_M_0->SetBinContent(21,77756.09439006035);
  S19_M_0->SetBinContent(22,59142.665732980306);
  S19_M_0->SetBinContent(23,56440.71592792033);
  S19_M_0->SetBinContent(24,44732.276772659745);
  S19_M_0->SetBinContent(25,45933.13668602021);
  S19_M_0->SetBinContent(26,44131.83681598023);
  S19_M_0->SetBinContent(27,36626.427357479835);
  S19_M_0->SetBinContent(28,42030.32696759977);
  S19_M_0->SetBinContent(29,36025.98740080032);
  S19_M_0->SetBinContent(30,28220.357963960167);
  S19_M_0->SetBinContent(31,23116.678332179985);
  S19_M_0->SetBinContent(32,29421.227877319914);
  S19_M_0->SetBinContent(33,27619.928007279934);
  S19_M_0->SetBinContent(34,19213.85861376027);
  S19_M_0->SetBinContent(35,24017.328267199973);
  S19_M_0->SetBinContent(36,21015.158483800245);
  S19_M_0->SetBinContent(37,20714.94850545977);
  S19_M_0->SetBinContent(38,17412.55874372029);
  S19_M_0->SetBinContent(39,14110.178981980083);
  S19_M_0->SetBinContent(40,12909.309068620338);
  S19_M_0->SetBinContent(41,288508.179184738); // overflow
  S19_M_0->SetEntries(10000);
  // Style
  S19_M_0->SetLineColor(9);
  S19_M_0->SetLineStyle(1);
  S19_M_0->SetLineWidth(1);
  S19_M_0->SetFillColor(9);
  S19_M_0->SetFillStyle(1001);

  // Creating a new THStack
  THStack* stack = new THStack("mystack_38","mystack");
  stack->Add(S19_M_0);
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
  stack->GetXaxis()->SetTitle("M [ l+_{1} p_{2} p_{3} ] (GeV/c^{2}) ");

  // Finalizing the TCanvas
  canvas->SetLogx(0);
  canvas->SetLogy(1);

  // Saving the image
  canvas->SaveAs("../../HTML/MadAnalysis5job_0/selection_18.png");
  canvas->SaveAs("../../PDF/MadAnalysis5job_0/selection_18.png");
  canvas->SaveAs("../../DVI/MadAnalysis5job_0/selection_18.eps");

}
