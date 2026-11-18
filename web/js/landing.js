// Landing page: fill the live numbers from the generated files (quant_params.js, model_int8.js,
// results.js). Nothing here is typed by hand, so the page cannot drift from the code.
(function () {
  "use strict";
  const RES = window.PROJECT_RESULTS || {};
  const Q = window.QUANT_PARAMS || {};

  document.querySelectorAll("[data-r]").forEach((el) => {
    let v = RES;
    for (const k of el.dataset.r.split(".")) { if (v == null) return; v = v[k]; }
    if (v == null) return;
    el.textContent = el.dataset.f === "mhz" ? Number(v).toFixed(1) : Number(v).toLocaleString("en");
  });

  document.querySelectorAll("[data-q]").forEach((el) => {
    if (el.dataset.q === "int8" && Q.int8TestAccuracy) el.textContent = Q.int8TestAccuracy.toFixed(2) + "%";
    if (el.dataset.q === "fp32" && window.FP32_TEST_ACCURACY) el.textContent = window.FP32_TEST_ACCURACY.toFixed(2) + "%";
  });

  const fw = RES.firmware;
  if (fw && fw.optMacInstructionsPerInference) {
    const x = (fw.optScalarInstructionsPerInference / fw.optMacInstructionsPerInference).toFixed(2) + "×";
    for (const id of ["map-speedup", "proof-speedup"]) {
      const el = document.getElementById(id);
      if (el) el.textContent = x;
    }
  }

  const soc = RES.soc, sw = RES.socSweep, ar = RES.socArea;
  const put = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
  if (soc && soc.optMacCycles) {
    put("soc-speedup", (soc.optScalarCycles / soc.optMacCycles).toFixed(2) + "×");
  }
  if (sw && sw.rows && sw.rows.length) {
    put("soc-speedup16", sw.rows[sw.rows.length - 1].speedup.toFixed(2) + "×");
  }
  if (soc && ar && ar.ecp5) {
    const fNo = Math.max(...ar.ecp5.coreNoMac.fmaxMHz), fMac = Math.max(...ar.ecp5.coreMac.fmaxMHz);
    const tNo = soc.optScalarCycles / fNo, tMac = soc.optMacCycles / fMac;
    put("soc-time", (tNo / tMac).toFixed(1) + "×");
    put("soc-fmax-drop", ((1 - fMac / fNo) * 100).toFixed(0));
  }
})();
