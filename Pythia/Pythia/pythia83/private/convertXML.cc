// convert.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.
// PHP-specific additions written by Ben Lloyd.

// Execute with "private/convertXML" from main directory.

// This program converts existing .xml files into .html and .php analogues.

//==========================================================================

// Stdlib header files for character manipulation.
#include <cctype>

// Stdlib header files for input/output.
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

// Generic utilities.
using std::tolower;
using std::swap;

// Standard containers.
using std::string;
using std::map;
using std::vector;
using std::iterator;

// Input/output streams.
using std::cin;
using std::cout;
using std::cerr;
using std::ios;
using std::istream;
using std::ostream;
using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::ostringstream;

// Input/output formatting.
using std::endl;

//==========================================================================

// Generic facilities.

//--------------------------------------------------------------------------

// Convert string to lowercase for case-insensitive comparisons.
// Also remove initial and trailing blanks, if any.

string toLower(const string& name) {

  // Copy string without initial and trailing blanks.
  if (name.find_first_not_of(" ") == string::npos) return "";
  int firstChar = name.find_first_not_of(" ");
  int lastChar  = name.find_last_not_of(" ");
  string temp   = name.substr( firstChar, lastChar + 1 - firstChar);

  // Convert to lowercase letter by letter.
  for (int i = 0; i < int(temp.length()); ++i)
    temp[i] = std::tolower(temp[i]);
  return temp;

}

//--------------------------------------------------------------------------

// Extract XML value string following XML attribute.

string attributeValue(string line, string attribute) {

  if (line.find(attribute) == string::npos) return "";
  int iBegAttri = line.find(attribute);
  int iBegQuote = line.find_first_of("\"'", iBegAttri + 1);
  int iEndQuote = line.find_first_of("\"'", iBegQuote + 1);
  return line.substr(iBegQuote + 1, iEndQuote - iBegQuote - 1);

}

//==========================================================================

// Classes and methods used to build an index of all documented methods,
// and to write a file with the relevant info.

//--------------------------------------------------------------------------

// Class with info on a method.
class MethodInfo {
public:
  MethodInfo(string methodIn = " ", string fileIn = " ",
    string anchorIn = " ") : method(methodIn), file(fileIn),
    anchor(anchorIn) {}
  string method, file, anchor;
};

//--------------------------------------------------------------------------

// Map of methods documented in the xml files.
map<string, MethodInfo> methodsMap;

//--------------------------------------------------------------------------

// Generate index.

bool constructMethods(string convType) {

  // Conversion to .html or to .php.
  bool toHTML = (convType == "html");
  bool toPHP  = (convType == "php");
  if (!toHTML && !toPHP) {
    cout << "Error: unknown conversion type " << convType << "\n";
    return false;
  }

  // Open output file.
  string nameOut = (toHTML) ? "share/Pythia8/htmldoc/ProgramMethods.html"
    : "share/Pythia8/phpdoc/ProgramMethods.php";
  const char* nameOutCstring = nameOut.c_str();
  ofstream os(nameOutCstring);
  if (!os) {
    cout << "Error: user update file " << nameOut << " not found \n";
    return false;
  }

  // Write header material of file.
  os << "<html>\n<head>\n<title>Program Methods</title>\n"
     << "<link rel=\"stylesheet\" type=\"text/css\" href=\"pythia.css\"/>\n"
     << "<link rel=\"shortcut icon\" href=\"pythia32.gif\"/>\n"
     << "</head>\n<body>\n\n<h2>Program Methods</h2>\n\n";
  os << "This is an alphabetical index of all methods that "
     << "are documented elsewhere\n"
     << "on these pages, beginning with the few methods that do not "
     << "belong to a class.\n"
     << "Many of them are only intended for experts, while undocumented "
     << "ones are only for code authors.\n<p/>\n";
  os << "<table cellspacing=\"5\">\n\n<tr>\n<td><b>Return type</b></td>\n"
     << "<td><b>Method name</b></td>\n<td><b>Documentation page</b></td>\n"
     << "</tr>\n\n";

  // Loop first time methods outside classes, second time in classes.
  for (int iClass = 0; iClass < 2; ++iClass) {

    // Loop through map of methods.
    map<string, MethodInfo>::iterator methodsEntry = methodsMap.begin();
    while (methodsEntry != methodsMap.end() ) {
      string fullMethod = methodsEntry->second.method;

      // Separate methods in and outside class.
      bool   inClass    = (fullMethod.find("::") != string::npos);
      if ( (iClass == 0 && !inClass) || (iClass == 1 && inClass) ) {

        // Begin separation into different fields.
        string attri    = " ";
        string mName    = fullMethod;
        string fileName = methodsEntry->second.file;
        string aName    = methodsEntry->second.anchor;

        // Separate return types from method name itself.
        if (fullMethod.rfind(" ") != string::npos) {
          int iBegProper = fullMethod.rfind(" ");
          attri = fullMethod.substr(0, iBegProper);
          mName.replace(0, iBegProper + 1, "");
        }

        // Turn method name into anchor.
        string anchor = "<a href=\"" + fileName + "." + convType
          + "#" + aName + "\" target=\"page\">" + mName + "</a>";

        // Split file name into individual words.
        string fileNameSep = fileName;
        string capitals    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        int    looked      = 1;
        if (fileName.substr(0,5) == "HepMC") looked = 5;
        while (fileNameSep.find_first_of( capitals, looked)
          != string::npos) {
          looked = fileNameSep.find_first_of( capitals, looked);
          fileNameSep.insert(looked, " ");
          looked += 2;
        }
        if (fileName.substr(0,4) == "Four") fileNameSep[4] = '-';

        // Print out info, embedded in tags.
        os << "<tr>\n<td>" << attri << "</td>\n<td>" << anchor
           << "</td>\n<td>" << fileNameSep << "</td>\n</tr>\n\n";
      }

      // End of loop through map of methods.
      ++methodsEntry;
    }
  }

  // Write footer material of file.
  os << "</table>\n\n<!-- Copyright (C) 2021 Torbjorn Sjostrand -->\n";

  // Done.
  return true;

}

//==========================================================================

// Convert from xml to html or php: the key method of this program.

bool convertFile(string nameRoot, string convType) {

  // Conversion to .html or to .php.
  cout << nameRoot << "\n";
  bool toHTML = (convType == "html");
  bool toPHP  = (convType == "php");
  if (!toHTML && !toPHP) {
    cout << "Error: unknown conversion type " << convType << "\n";
    return false;
  }

  // Check input file exists
  string nameIn = "share/Pythia8/xmldoc/" + nameRoot + ".xml";
  const char* nameInCstring = nameIn.c_str();
  ifstream isCheck(nameInCstring);
  if (!isCheck) {
    cout << "Error: user update file " << nameIn << " not found \n";
    return false;
  }

  // Read input file and store as vector<string>.
  // Check for <h3> tags outside comments (for tableOfContents).
  bool makeTOC = false;
  int  iRead   = 0;
  string line;
  vector<string> lines;
  vector<string> h3titles;
  bool insideComment = false;
  while ( getline(isCheck, line) ) {
    lines.push_back(line);
    if (line.find("<!--") != string::npos) insideComment = true;
    if (insideComment) {
      if (line.find("-->") != string::npos) insideComment = false;
      continue;
    }
    ++iRead;
    if (line.find("<h3") != string::npos) {
      // Save title of this <h3> section.
      int iBeg = line.find("<h3");
      iBeg = line.find(">", iBeg) + 1;
      int iEnd = line.find("</h3", iBeg);
      string h3tit = line.substr(iBeg,iEnd - iBeg);
      h3titles.push_back(h3tit);
      // Add anchor for use by TOC. Numbered sequentially.
      ostringstream osNum;
      osNum << h3titles.size() - 1;
      string anchorLine = "<a name=\"section" + osNum.str() + "\"></a> \n";
      lines[lines.size()-1] = anchorLine + line;
      // Only make table of contents for large files with several <h3> tags.
      if (iRead >= 80 && h3titles.size() >= 2) makeTOC = true;
    }
  }
  // Done with input file (now stored in vector<string> lines buffer).
  isCheck.close();

  // Open output file.
  string nameOut = (toHTML) ? "share/Pythia8/htmldoc/" + nameRoot + ".html"
    : "share/Pythia8/phpdoc/" + nameRoot + ".php";
  const char* nameOutCstring = nameOut.c_str();
  ofstream os(nameOutCstring);
  if (!os) {
    cout << "Error: user update file " << nameOut << " not found \n";
    return false;
  }

  // Current anchor for the tag.
  int anchorNumber = 0;

  // Number of settings picked up.
  int settingnum = 0;
  vector<string> names;
  vector<string> defaultnames;

  // Read in one line at a time.
  int searchEndPDT = 0;
  int iLine = -1;
  int nLines = lines.size();
  while (++iLine < nLines) {

    string line = lines[iLine];

    // Everything should be unchanged inside comment. For simplicity
    // assume that comments are on separate lines, i.e. not inside text.
    if (line.find("<!--") != string::npos) insideComment = true;
    if (insideComment) {
      os << line << endl;
      if (line.find("-->") != string::npos) insideComment = false;
      continue;
    }

    // PHP: This following if statement is to insert a huge PHP script
    // to deal with the file opening, permissions, closing and deleting.
    // Do not amend unless you know php!
    if (toPHP && line.find("<PHPFILECODE/>") != string::npos) {
      // Open php input file.
      string phpIn = "share/Pythia8/phpdoc/php.txt";
      const char* phpInCstring = phpIn.c_str();
      ifstream isphp(phpInCstring);
      if (!isphp) {
        cout << "Error: php content file " << phpIn << " not found \n";
        return false;
      }
      // Write line by line from input file to normal output.
      string phpline;
      while ( getline(isphp, phpline) ) {
        os << phpline << endl;
      }
      continue;
    }

    // PHP: Replace chapter tokens, preserving chapter name.
    // Welcome page should not have body, only contain frames.
    if (toPHP && line.find("<chapter") != string::npos) {
      int iBeg = line.find("\"") + 1;
      int iEnd = line.rfind("\"");
      string chapterName = line.substr(iBeg, iEnd - iBeg);
      os << "<html>" << endl << "<head>" << endl << "<title>"
         << chapterName << "</title>" << endl
         // New: cascading style sheet information and Pythia logo.
         << "<link rel=\"stylesheet\" type=\"text/css\" "
         << "href=\"pythia.css\"/>" << endl
         << "<link rel=\"shortcut icon\" href=\"pythia32.gif\"/>" << endl
         << "</head>" << endl;
      // Special info on top of the SaveSettings page.
      if(nameRoot == "SaveSettings")
      os << endl << "<?php" << endl << "if($_GET[\"filename\"] != \"\")"
         << endl << "{" << endl
         << "$_POST[\"filename\"] = $_GET[\"filename\"];"
         << endl << "}" << endl << "?>" << endl
         << "<script language=\"JavaScript1.2\">" << endl
         << "<?php echo \"setTimeout(\\\"top.index.location='Index.php?"
         << "filename=\".$_POST[\"filename\"].\"'\\\", 1)\"; ?>" << endl
         << "</script>" << endl;
      if (nameRoot != "Welcome")
        os  << "<body>" << endl << endl
            << "<script language=javascript type=text/javascript>"
            << endl << "function stopRKey(evt) {" << endl
            << "var evt = (evt) ? evt : ((event) ? event : null);"
            << endl << "var node = (evt.target) ? evt.target :"
            << "((evt.srcElement) ? evt.srcElement : null);" << endl
            << "if ((evt.keyCode == 13) && (node.type==\"text\"))"
            << endl << "{return false;}" << endl << "}" << endl << endl
            << "document.onkeypress = stopRKey;" << endl
            << "</script>" << endl << "<?php" << endl
            << "if($_POST['saved'] == 1) {" << endl
            << "if($_POST['filepath'] != \"files/\") {" << endl
            << "echo \"<font color='red'>SETTINGS SAVED TO FILE</font>"
            << "<br/><br/>\"; }" << endl << "else {" << endl
            << "echo \"<font color='red'>NO FILE SELECTED YET.. PLEASE DO SO"
            << " </font><a href='SaveSettings.php'>HERE</a><br/><br/>\"; }"
            << endl << "}" << endl << "?>" << endl << endl
            << "<form method=\'post\' action=\'" << nameRoot
            << ".php\'>" << endl;
      continue;
    }

    // PHP: Replace end-of-chapter token with save-button if > 0 to save.
    if (toPHP && line.find("</chapter") != string::npos) {
      if(settingnum > 0) {
        os << "<input type=\"hidden\" name=\"saved\" value=\"1\"/>" << endl
           << endl << "<?php" << endl << "echo \"<input type='hidden'"
           << " name='filepath' value='\".$_GET[\"filepath\"].\"'/>\"?>"
           << endl << endl
           << "<table width=\"100%\"><tr><td align=\"right\"><input "
           << "type=\"submit\" value=\"Save Settings\" /></td></tr></table>"
           << endl << "</form>" << endl << endl << "<?php" << endl << endl
           << "if($_POST[\"saved\"] == 1)" << endl << "{" << endl
           << "$filepath = $_POST[\"filepath\"];" << endl
           << "$handle = fopen($filepath, 'a');" << endl << endl;
        int i = 0;
        for (i = 1; i < settingnum+1; i++)
          os <<  "if($_POST[\"" << i << "\"] != \"" << defaultnames[i-1]
             << "\")" << endl << "{" << endl << "$data = \"" << names[i-1]
             << " = \".$_POST[\"" << i << "\"].\"\\n\";" << endl
             << "fwrite($handle,$data);" << endl << "}" << endl;
        os << "fclose($handle);" << endl << "}" << endl << endl
           << "?>" << endl;
      }
      if (nameRoot != "Welcome") os << "</body>" << endl;
      os << "</html>" << endl;
      continue;
    }

    // HTML: Replace chapter tokens, preserving chapter name.
    // Welcome page should not have body, only contain frames.
    if (line.find("<chapter") != string::npos) {
      int iBeg = line.find("\"") + 1;
      int iEnd = line.rfind("\"");
      string chapterName = line.substr(iBeg, iEnd - iBeg);
      os << "<html>" << endl << "<head>" << endl << "<title>"
         << chapterName << "</title>" << endl
         // New: cascading style sheet information and Pythia logo.
         << "<link rel=\"stylesheet\" type=\"text/css\" "
         << "href=\"pythia.css\"/>" << endl
         << "<link rel=\"shortcut icon\" href=\"pythia32.gif\"/>" << endl
         << "</head>" << endl;
      if (nameRoot != "Welcome") os  << "<body>" << endl;
      continue;
    }
    if (line.find("</chapter") != string::npos) {
      if (nameRoot != "Welcome") os << "</body>" << endl;
      os << "</html>" << endl;
      continue;
    }

    // PHP: Sections of index that allow save operations.
    if (toPHP && line.find("<INDEXPHP>") != string::npos) {
      os << "<?php" << endl << "$filepath = \"files/\".$_GET[\"filename\"];"
         << endl << "$filename = $_GET[\"filename\"];" << endl << "echo \"";
      continue;
    }
    if (toPHP && line.find("</INDEXPHP>") != string::npos) {
      os << endl << "\";?>" << endl;
      continue;
    }

    // HTML: Remove tags used specifically for PHP code.
    if (line.find("<PHPFILECODE/>") != string::npos) continue;
    if (line.find("<INDEXPHP>") != string::npos) continue;
    if (line.find("</INDEXPHP>") != string::npos) continue;

    // Links to pages from index.
    if (line.find("<aidx") != string::npos) {
      int iBeg = line.find("<aidx");
      int iEnd = line.find(">", iBeg);
      string href = attributeValue(line, "href");
      string linetmp = line.substr(0, iBeg);
      if (toHTML) linetmp += "<a href=\"" + href
         + ".html\" target=\"page\">";
      else if (href == "SaveSettings") linetmp += "<a href='" + href
         + ".php?returning=1&filename=\".$filename.\"' target='page'>";
      else linetmp += + "<a href='" + href
         + ".php?filepath=\".$filepath.\"' target='page'>";
      linetmp += line.substr(iEnd + 1, line.length() - iEnd);
      line = linetmp;
    }

    // End of links from index.
    while (line.find("</aidx>") != string::npos)
      line.replace( line.find("</aidx>"), 7, "</a>");

    // Local links between pages in online manual.
    while (line.find("<aloc") != string::npos) {
      int iBeg = line.find("<aloc");
      int iEnd = line.find(">", iBeg);
      string href = attributeValue(line, "href");
      string linetmp = line.substr(0, iBeg);
      if (toHTML) linetmp += "<a href=\"" + href
        + ".html\" target=\"page\">";
      else linetmp += "<?php $filepath = $_GET[\"filepath\"];\n"
        "echo \"<a href='" + href
        + ".php?filepath=\".$filepath.\"' target='page'>\";?>";
      linetmp += line.substr(iEnd + 1, line.length() - iEnd);
      line = linetmp;
    }

    // End of local links.
    while (line.find("</aloc>") != string::npos)
      line.replace( line.find("</aloc>"), 7, "</a>");

    // Replace anchors to .xml files by ones to .html/.php in Welcome.xml.
    if(nameRoot == "Welcome") {
      if (toHTML) while (line.find(".xml") != string::npos)
        line.replace( line.find(".xml"), 4, ".html");
      if (toPHP)  while (line.find(".xml") != string::npos)
        line.replace( line.find(".xml"), 4, ".php");
    }

    // PHP: Interactive parameters with fill-in boxes.
    if (toPHP && (line.find("<parm ") != string::npos
               || line.find("<pvec ") != string::npos)) {
      // Check for continuation line.
      if (line.rfind(">") == string::npos && ++iLine < nLines)
        line += lines[iLine];
      string name = attributeValue(line, "name=");
      string defaultname = attributeValue(line, "default=");
      string min = attributeValue(line, "min=");
      string max = attributeValue(line, "max=");
      bool makeBracket = (defaultname != "" || min != "" || max != "");
      settingnum++;
      names.push_back(name);
      defaultnames.push_back(defaultname);
      os << "<br/><br/><table><tr><td><strong>" << name
         << " </td><td></td><td> <input type=\"text\" name=\""
         << settingnum << "\" value=\"" << defaultname
         << "\" size=\"20\"/> ";
      if (makeBracket) os << " &nbsp;&nbsp;(";
      if (defaultname != "") os << "<code>default = <strong>"
        << defaultname << "</strong></code>";
      if (min != "") os << "; <code>minimum = " << min << "</code>";
      if (max != "") os << "; <code>maximum = " << max << "</code>";
      if (makeBracket) os << ")";
      os << "</td></tr></table>" << endl;
      continue;
    }

    // PHP: Interactive flags with radio buttons or fill-in boxes.
    if (toPHP && ( (line.find("<flag") != string::npos)
      || (line.find("<fvec") != string::npos) ) ) {
      // Check for continuation line.
      if (line.rfind(">") == string::npos && ++iLine < nLines)
        line += lines[iLine];
      string name = attributeValue(line, "name=");
      string defaultname = attributeValue(line, "default=");
      bool makeBracket = (defaultname != "");

      // PHP: Flag vector with fill-in boxes.
      if (line.find("<fvec") != string::npos) {
        settingnum++;
        names.push_back(name);
        defaultnames.push_back(defaultname);
        os << "<br/><br/><table><tr><td><strong>" << name
           << "  </td><td></td><td> <input type=\"text\" name=\""
           << settingnum << "\" value=\"" << defaultname
           << "\" size=\"20\"/> ";
        if (makeBracket) os << " &nbsp;&nbsp;(";
        if (defaultname != "") os << "<code>default = <strong>"
          << defaultname << "</strong></code>";
        if (makeBracket) os << ")";
        os << "</td></tr></table>" << endl;
        continue;
      }

      // PHP: Flags with radio buttons.
      else {
        settingnum++;
        names.push_back(name);
        defaultnames.push_back(defaultname);
        os << "<br/><br/><strong>" << name
           << "</strong>  <input type=\"radio\" name=\"" << settingnum
           << "\" value=\"on\"";
        if (defaultname == "on") os << " checked=\"checked\">";
        else os << ">";
        os << "<strong>On</strong>" << endl;
        os << "<input type=\"radio\" name=\"" << settingnum
           << "\" value=\"off\"";
        if (defaultname == "off") os << " checked=\"checked\">";
        else os << ">";
        os << "<strong>Off</strong>" << endl;
        if (defaultname != "")  os << " &nbsp;&nbsp;(<code>default = <strong>"
                                   << defaultname << "</strong></code>)";
        os << "<br/>" << endl;
        continue;
      }
    }

    // PHP: Interactive modes and words: begin common identification.
    if (toPHP && ( (line.find("<modeopen") != string::npos)
      || (line.find("<modepick") != string::npos)
      || (line.find("<word") != string::npos)
      || (line.find("<mvec") != string::npos) ) ) {
      // Check for continuation line.
      if (line.rfind(">") == string::npos && ++iLine < nLines)
        line += lines[iLine];
      string name = attributeValue(line, "name=");
      string defaultname = attributeValue(line, "default=");
      string min = attributeValue(line, "min=");
      string max = attributeValue(line, "max=");
      bool makeBracket = (defaultname != "" || min != "" || max != "");

      // PHP: Modes and words with fill-in boxes.
      if ( (line.find("<modeopen") != string::npos)
        || (line.find("<word") != string::npos)
        || (line.find("<mvec") != string::npos) ) {
        settingnum++;
        names.push_back(name);
        defaultnames.push_back(defaultname);
        os << "<br/><br/><table><tr><td><strong>" << name
           << "  </td><td></td><td> <input type=\"text\" name=\""
           << settingnum << "\" value=\"" << defaultname
           << "\" size=\"20\"/> ";
        if (makeBracket) os << " &nbsp;&nbsp;(";
        if (defaultname != "") os << "<code>default = <strong>"
          << defaultname << "</strong></code>";
        if (min != "") os << "; <code>minimum = " << min << "</code>";
        if (max != "") os << "; <code>maximum = " << max << "</code>";
        if (makeBracket) os << ")";
        os << "</td></tr></table>" << endl;
        continue;
      }

      // PHP: Modes with radio buttons.
      else if (line.find("<modepick") != string::npos) {
        string value = "";
        string desc = "";
        settingnum++;
        names.push_back(name);
        defaultnames.push_back(defaultname);
        os << "<br/><br/><table><tr><td><strong>" << name
           << "  </td><td> ";
        if (makeBracket) os << " &nbsp;&nbsp;(";
        if (defaultname != "") os << "<code>default = <strong>"
          << defaultname << "</strong></code>";
        if (min != "") os << "; <code>minimum = " << min << "</code>";
        if (max != "") os << "; <code>maximum = " << max << "</code>";
        if (makeBracket) os << ")";
        os << "</td></tr></table>" << endl;
        // First option found.
        if (++iLine < nLines) line = lines[iLine];
        while (line.find("<option") == string::npos && ++iLine < nLines) {
          os << line << endl;
          line = lines[iLine];
        }
        os << "<br/>" << endl;
        while (line.find("</modepick>") == string::npos) {
          if (line.find("value=") != string::npos) {
            int beg = line.find("value=");
            int end = line.find("\"", beg + 8);
            value = line.substr(beg + 7, end - beg - 7);
            if (line.find("</option>") != string::npos)  {
              int lenlab = beg + value.length() + 9;
              desc = line.substr( lenlab, line.find("</option>") - lenlab);
            }
            else {
              desc = line.substr(line.find(value)+ value.length()+2,
                     line.length()-(line.find(value)+ value.length()+2));
              if (++iLine < nLines) line = lines[iLine];
              while (line.find("</option>") == string::npos
                && ++iLine < nLines) {
                desc = desc + " " + line;
                line = lines[iLine];
              }
              desc = desc + " " + (line.substr(0, (line.find("</option"))));
            }
            //Point to add new option desc + value set.
            os << "<input type=\"radio\" name=\"" << settingnum
               << "\" value=\"" << value << "\"";
            if (value == defaultname) os << " checked=\"checked\">";
            else os << ">";
            os << "<strong>" << value << " </strong>: " << desc
               << "<br/>" << endl;

          // Output lines not part of option. Limited conversion.
          } else {
            while (line.find("<note>") != string::npos)
              line.replace( line.find("<note>"), 6, "<br/><b>");
            while (line.find("</note>") != string::npos)
              line.replace( line.find("</note>"), 7, "</b>");
            while (line.find("<notenl>") != string::npos)
              line.replace( line.find("<notenl>"), 8, "<b>");
            while (line.find("</notenl>") != string::npos)
              line.replace( line.find("</notenl>"), 9, "</b>");
            os << line << endl;
          }
          if (++iLine < nLines) line = lines[iLine];
        }

        // Go to next line.
        continue;
      }
    }

    // Identify several begintags for new section.
    // Many of these already covered above for PHP, but also there
    // some leftover cases (e.g. flagfix, modefix, parmfix, wordfix).
    string tagVal = "void";
    if ( line.find("<flag") != string::npos) tagVal = "flag";
    if ( line.find("<mode") != string::npos) tagVal = "mode";
    if ( line.find("<parm") != string::npos) tagVal = "parm";
    if ( line.find("<word") != string::npos) tagVal = "word";
    if ( line.find("<fvec") != string::npos) tagVal = "fvec";
    if ( line.find("<mvec") != string::npos) tagVal = "mvec";
    if ( line.find("<pvec") != string::npos) tagVal = "pvec";
    if ( line.find("<wvec") != string::npos) tagVal = "wvec";
    if ( line.find("<file") != string::npos) tagVal = "file";
    if ( line.find("<class") != string::npos) tagVal = "class";
    if ( line.find("<method") != string::npos) tagVal = "method";
    if (tagVal != "void") {
      // Check for continuation lines.
      while (line.rfind(">") == string::npos && ++iLine < nLines)
        line += lines[iLine];
      // Extract extra qualifiers.
      int iBeg = line.find("<" + tagVal);
      string nameVal = attributeValue(line, "name");
      string defVal  = attributeValue(line, "default=");
      string minVal  = attributeValue(line, "min=");
      string maxVal  = attributeValue(line, "max=");
      bool makeBracket = (defVal != "" || minVal != "" || maxVal != "");
      bool isMore    = (line.substr(iBeg + 1 + tagVal.size(), 4) == "more");
      // Write back info, suitably formatted.
      string linetmp = line.substr(0, iBeg);
      if (!isMore) linetmp += "<p/>";
      if (tagVal != "method") linetmp += "<code>" + tagVal + "&nbsp; </code>";
      linetmp += "<strong> " + nameVal + " &nbsp;</strong> ";
      if (makeBracket)  linetmp += "\n (";
      if (defVal != "") linetmp += "<code>default = <strong>"
        + defVal + "</strong></code>";
      if (minVal != "") linetmp += "; <code>minimum = "
        + minVal + "</code>";
      if (maxVal != "") linetmp += "; <code>maximum = "
        + maxVal + "</code>";
      if (makeBracket)  linetmp += ")";
      linetmp += "<br/>";
      line = linetmp;

      // Insert anchor for the tag.
      ++anchorNumber;
      ostringstream oNumber;
      oNumber << anchorNumber;
      string anchorTag = "anchor" + oNumber.str();
      os << "<a name=\"" + anchorTag + "\"></a>" << endl;

      // Fill map of all methods and insert anchors to them.
      if (tagVal == "method") {

        // Replace arguments list with ... to make more compact.
        string methodsTmp = nameVal;
        if (methodsTmp.find("(", 0) != string::npos) {
          int iBegMethods = methodsTmp.find("(", 0);
          if (methodsTmp.find(")", iBegMethods+1)) {
            int iEndMethods = methodsTmp.find(")", iBegMethods+1);
            if (iEndMethods - iBegMethods > 1) methodsTmp.replace(
              iBegMethods + 1, iEndMethods - iBegMethods - 1, "...");
          }
        }
        string methodsVal = methodsTmp;
        // Remove return type for alphabetical ordering.
        if (methodsTmp.rfind(" ") != string::npos) {
          int iBegProper = methodsTmp.rfind(" ");
          methodsTmp.replace(0, iBegProper + 1, "");
        }
        string methodsTag = toLower(methodsTmp);
        // Add star when tag already exists.
        bool stillAdding = true;
        while (stillAdding) {
          if (methodsMap.count(methodsTag) > 0) methodsTag += "*";
          else stillAdding = false;
        }
        // Insert entry into map.
        methodsMap[methodsTag] = MethodInfo(methodsVal, nameRoot,
          anchorTag);
      }
    }

    // Identify one kind of begintags for new line.
    tagVal = "void";
    if ( line.find("<argument") != string::npos) tagVal = "argument";
    if (tagVal != "void") {
      // Extract extra qualifiers.
      int iBeg = line.find("<" + tagVal);
      int iEnd = line.find(">", iBeg);
      string nameVal = attributeValue(line, "name");
      string defVal = attributeValue(line, "default");
      // Write back info, suitably formatted.
      string linetmp = line.substr(0, iBeg)
        + "<br/><code>" + tagVal + "</code><strong> "
        + nameVal + " </strong> ";
      if (defVal != "") linetmp += "(<code>default = <strong>"
        + defVal + "</strong></code>)";
      linetmp += " : " + line.substr(iEnd + 1, line.length() - iEnd);
      line = linetmp;
    }

    // Identify other kinds of begintags for new line.
    tagVal = "void";
    if ( line.find("<option") != string::npos) tagVal = "option";
    if ( line.find("<argoption") != string::npos) tagVal = "argoption";
    if (tagVal != "void") {
      // Extract extra qualifiers.
      int iBeg = line.find("<" + tagVal);
      int iEnd = line.find(">", iBeg);
      if (tagVal == "argoption") tagVal = "argumentoption";
      string valVal = attributeValue(line, "value");
      // Write back info, suitably formatted.
      string linetmp = line.substr(0, iBeg)
        + "<br/><code>" + tagVal + " </code><strong> "
        + valVal + "</strong> : "
        + line.substr(iEnd + 1, line.length() - iEnd);
      line = linetmp;
    }


    // Replace <refit>...</refit> by <dt>...</dt> in Bibliography.xml
    // and also insert an anchor.
    int tbeg = 0;
    while ( (tbeg = line.find("<refit>")) != string::npos) {
      int tend = line.find("</refit>");
      string name = line.substr(tbeg + 7, tend - tbeg - 7);
      string replacement = "<dt><a name=\"ref" + name + "\">" + name
        + "</a></dt>";
      line.replace(tbeg, tend - tbeg + 8, replacement);
    }

    // Replace <ref>...</ref> by [...] for references,
    // and additionally make them anchors to the correct place in the
    // reference section.
    while ( (tbeg = line.find("<ref>")) != string::npos) {
      string replacement = "[";
      int tend = line.find("</ref>");
      string ref;
      for ( int pos = tbeg + 5; pos < tend + 1; ++pos ) {
        if ( line[pos] == ' ' || line[pos] == ',' || line[pos] == '<' ) {
          if ( !ref.empty() ) {
            if (toHTML)
              replacement += "<a href=\"Bibliography.html";
            if (toPHP)
              replacement += "<a href=\"Bibliography.php";
            replacement += "#ref" + ref + "\" target=\"page\">" + ref + "</a>";
          }
          if ( line[pos] != '<' ) replacement += line[pos];
          ref = "";
        } else
          ref += line[pos];
      }
      replacement += ']';
      line.replace(tbeg, tend - tbeg + 6, replacement);
    }

    // Replace equations by italics.
    while (line.find("<ei>") != string::npos)
      line.replace( line.find("<ei>"), 4, "<i>");
    while (line.find("</ei>") != string::npos)
      line.replace( line.find("</ei>"), 5, "</i>");
    while (line.find("<eq>") != string::npos)
      line.replace( line.find("<eq>"), 4, "<br/><i>");
    while (line.find("</eq>") != string::npos)
      line.replace( line.find("</eq>"), 5, "</i><br/>");

    // Replace note by boldface, with or without new line.
    while (line.find("<note>") != string::npos)
      line.replace( line.find("<note>"), 6, "<br/><b>");
    while (line.find("</note>") != string::npos)
      line.replace( line.find("</note>"), 7, "</b>");
    while (line.find("<notenl>") != string::npos)
      line.replace( line.find("<notenl>"), 8, "<b>");
    while (line.find("</notenl>") != string::npos)
      line.replace( line.find("</notenl>"), 9, "</b>");

    // Replace unused endtags by empty space.
    while (line.find("more>") != string::npos
      && line.find("</") != string::npos)
      line.erase( line.find("more>"), 4);
    while (line.find("</flag>") != string::npos)
      line.replace( line.find("</flag>"), 7, "  ");
    while (line.find("</mode>") != string::npos)
      line.replace( line.find("</mode>"), 7, "  ");
    while (line.find("</modeopen>") != string::npos)
      line.replace( line.find("</modeopen>"), 11, "  ");
    while (line.find("</modepick>") != string::npos)
      line.replace( line.find("</modepick>"), 11, "  ");
    while (line.find("</modefix>") != string::npos)
      line.replace( line.find("</modefix>"), 10, "  ");
    while (line.find("</parm>") != string::npos)
      line.replace( line.find("</parm>"), 7, "  ");
    while (line.find("</parmfix>") != string::npos)
      line.replace( line.find("</parmfix>"), 10, "  ");
    while (line.find("</word>") != string::npos)
      line.replace( line.find("</word>"), 7, "  ");
    while (line.find("</fvec>") != string::npos)
      line.replace( line.find("</fvec>"), 7, "  ");
    while (line.find("</fvecfix>") != string::npos)
      line.replace( line.find("</fvecfix>"), 10, "  ");
    while (line.find("</mvec>") != string::npos)
      line.replace( line.find("</mvec>"), 7, "  ");
    while (line.find("</mvecfix>") != string::npos)
      line.replace( line.find("</mvecfix>"), 10, "  ");
    while (line.find("</pvec>") != string::npos)
      line.replace( line.find("</pvec>"), 7, "  ");
    while (line.find("</wvec>") != string::npos)
      line.replace( line.find("</wvec>"), 7, "  ");
    while (line.find("</pvecfix>") != string::npos)
      line.replace( line.find("</pvecfix>"), 10, "  ");
    while (line.find("</file>") != string::npos)
      line.replace( line.find("</file>"), 7, "  ");
    while (line.find("</class>") != string::npos)
      line.replace( line.find("</class>"), 8, "  ");
    while (line.find("</method>") != string::npos)
      line.replace( line.find("</method>"), 9, "  ");
    while (line.find("</argument>") != string::npos)
      line.replace( line.find("</argument>"), 11, "  ");
    while (line.find("</argoption>") != string::npos)
      line.replace( line.find("</argoption>"), 12, "  ");
    while (line.find("</option>") != string::npos)
      line.replace( line.find("</option>"), 9, "  ");

    // Remove </particle> lines.
    if (line.find("</particle>") != string::npos) continue;

    // Replace particle table tags by simple text.
    if (line.find("<particle ") != string::npos) {
      line.replace( line.find("<particle "), 10, "<p/>particle: ");
      ++searchEndPDT;
    }
    if (line.find("<channel ") != string::npos) {
      line.replace( line.find("<channel "), 9, "<br/>      channel: ");
      ++searchEndPDT;
    }

    // Search for endtags if in PDT environment.
    if (searchEndPDT > 0 && line.rfind("/>") != string::npos
      && line.rfind("/>") > 4) {
      line.erase( line.rfind("/>"), 2 );
      --searchEndPDT;
    }
    if (searchEndPDT > 0 && line.rfind(">") != string::npos
      && line.rfind("/>") > 4) {
      line.erase( line.rfind(">"), 1 );
      --searchEndPDT;
    }

    // Default case: write original or modified line to output.
    os << line << endl;

    // HTML: insert table of contents after <h2> (or <h1>) tag
    if ((line.find("</h2>") != string::npos
      || line.find("</h1>") != string::npos) && makeTOC) {
      os << "<ol id=\"toc\">" << endl;
      for (unsigned int iH3 = 0; iH3 < h3titles.size(); ++iH3) {
        os << "  <li><a href=\"#section" << iH3 <<"\">" << h3titles[iH3]
           << "</a></li>" << endl;
      }
      os << "</ol>" << endl << endl;
      makeTOC = false;
    }

    // End of loop over lines in file. Done.
  }


  return true;
}

//==========================================================================

// Convert files from xml to html and php one at a time.

int main() {

  // Flags for what to do.
  bool convert2HTML = true;
  bool convert2PHP = false;

  // Convert from xml to html.
  if (convert2HTML) {
    methodsMap.clear();

    // These two file names do not appear in the Index.xml file.
    convertFile("Welcome", "html");
    convertFile("Index", "html");

    // Extract other file names from the Index.xml file.
    ifstream is("share/Pythia8/xmldoc/Index.xml");
    string line;
    while ( getline(is, line) ) {
      if (line.find("<aidx") != string::npos) {
        string fileName = attributeValue( line, "href");
        // ProgramMethods file built by constructMethods below.
	// ExampleKeywords built by its own python script.
        if (fileName != "ProgramMethods" && 
          fileName != "ExampleKeywords")
          convertFile(fileName, "html");
      }
    }

    // Construct index of methods.
    constructMethods("html");
  }

  // Convert from xml to php.
  if (convert2PHP) {
    methodsMap.clear();

    // These two file names do not appear in the Index.xml file.
    convertFile("Welcome", "php");
    convertFile("Index", "php");

    // Extract other file names from the Index.xml file.
    ifstream is("share/Pythia8/xmldoc/Index.xml");
    string line;
    while ( getline(is, line) ) {
      if (line.find("<aidx") != string::npos) {
        string fileName = attributeValue( line, "href");
        // ProgramMethods file built by constructMethods below.
	// ExampleKeywords built by its own python script.
        if (fileName != "ProgramMethods" && 
          fileName != "ExampleKeywords")
          convertFile(fileName, "php");
      }
    }

    // Construct index of methods.
    constructMethods("php");
  }

  // Done.
  return 0;
}

//==========================================================================
