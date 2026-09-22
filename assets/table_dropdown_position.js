/* ---------------------------------------------------------------------------
   DataTable dropdown menus escape the table.

   A Dash DataTable scrolls its own body, so an in-cell dropdown opened on one
   of the lower rows is clipped by the bottom edge of the table -- no z-index
   helps, because the clipping is an overflow, not a stacking, problem. This
   re-anchors the open menu with position: fixed against the viewport, and
   flips it above the cell when there is more room up than down.

   The menu node is left where React put it, so react-select keeps its own
   click handling; only its geometry is taken over here.
   --------------------------------------------------------------------------- */
(function () {
  "use strict";

  var MENU_SELECTOR = ".Select-menu-outer";
  var MAX_MENU_HEIGHT = 320;
  var MIN_MENU_HEIGHT = 120;
  var MIN_MENU_WIDTH = 220;
  var GUTTER = 6;

  var open = { menu: null, control: null };
  var frame = null;

  // Re-measuring forces a reflow, so coalesce the scroll and resize bursts
  // into one placement per frame.
  function schedule() {
    if (frame === null) {
      frame = window.requestAnimationFrame(function () {
        frame = null;
        place();
      });
    }
  }

  function setImportant(el, styles) {
    Object.keys(styles).forEach(function (prop) {
      el.style.setProperty(prop, styles[prop], "important");
    });
  }

  function place() {
    var menu = open.menu;
    var control = open.control;

    if (!menu || !control || !menu.isConnected || !control.isConnected) {
      release();
      return;
    }

    var rect = control.getBoundingClientRect();
    var inner = menu.querySelector(".Select-menu");

    // Measure the natural height before constraining it again. react-select
    // caps the inner list as well, so both have to be released to measure.
    menu.style.setProperty("max-height", "none", "important");
    if (inner) {
      inner.style.setProperty("max-height", "none", "important");
    }
    var natural = Math.min(menu.scrollHeight + 2, MAX_MENU_HEIGHT);

    var roomBelow = window.innerHeight - rect.bottom - GUTTER;
    var roomAbove = rect.top - GUTTER;
    var openUp = roomBelow < Math.min(natural, MIN_MENU_HEIGHT) && roomAbove > roomBelow;
    var height = Math.max(Math.min(natural, openUp ? roomAbove : roomBelow), 0);

    var width = Math.max(rect.width, MIN_MENU_WIDTH);
    var left = Math.min(rect.left, Math.max(window.innerWidth - width - GUTTER, GUTTER));

    setImportant(menu, {
      position: "fixed",
      top: (openUp ? rect.top - height : rect.bottom) + "px",
      left: left + "px",
      width: width + "px",
      "max-height": height + "px",
      bottom: "auto",
      right: "auto",
      "z-index": "10000",
      "overflow-y": "auto",
    });

    if (inner) {
      setImportant(inner, { "max-height": height + "px" });
    }
  }

  function release() {
    open.menu = null;
    open.control = null;
    window.removeEventListener("scroll", schedule, true);
    window.removeEventListener("resize", schedule);
  }

  function capture(menu) {
    var select = menu.closest(".Select");
    var control = select && select.querySelector(".Select-control");
    if (!control) {
      return;
    }

    open.menu = menu;
    open.control = control;
    window.addEventListener("scroll", schedule, true);
    window.addEventListener("resize", schedule);
    place();
  }

  function scan(node, onFound) {
    if (node.nodeType !== 1) {
      return;
    }
    if (node.matches(MENU_SELECTOR)) {
      onFound(node);
      return;
    }
    var found = node.querySelector(MENU_SELECTOR);
    if (found) {
      onFound(found);
    }
  }

  new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
      Array.prototype.forEach.call(mutation.removedNodes, function (node) {
        scan(node, function (menu) {
          if (menu === open.menu) {
            release();
          }
        });
      });

      Array.prototype.forEach.call(mutation.addedNodes, function (node) {
        scan(node, function (menu) {
          if (menu.closest(".dash-table-container")) {
            capture(menu);
          }
        });
      });
    });
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
