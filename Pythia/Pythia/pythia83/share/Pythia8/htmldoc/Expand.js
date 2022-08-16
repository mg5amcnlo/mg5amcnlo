// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.
// Author: Philip Ilten, October 2019.

// Script to expand/collapse arrow buttons.
var expand = document.getElementsByClassName("expand");
var i;
for (i = 0; i < expand.length; i++) {
    expand[i].addEventListener("click", function() {
	this.classList.toggle("active");
	var panel = this.nextElementSibling;
	if (panel.style.display === "block") {
	    panel.style.display = "none";
	} else {
	    panel.style.display = "block";
	}
    });
}

// Script to expand/collapse all arrow buttons.
var expandAll = document.getElementsByClassName("expandAll");
var i;
for (i = 0; i < expandAll.length; i++) {
    expandAll[i].addEventListener("click", function() {
	var activeAll = this.classList.toggle("activeAll");
	for (i = 0; i < expand.length; i++) {
	    var panel = expand[i].nextElementSibling;
	    if (activeAll) {
		expand[i].classList.add("active");
		panel.style.display = "block";
	    } else {
		expand[i].classList.remove("active");
		panel.style.display = "none";
	    }
	}
    });
}

