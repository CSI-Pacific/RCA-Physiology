/* ---------------------------------------------------------------------------
   DataTable cells edit like ordinary text boxes when clicked.

   A Dash DataTable cell has two states. Selected (what a click gives you):
   the whole value is highlighted, typing replaces it and the arrow keys move
   to the next cell. Editing (normally reached by double-clicking): a caret,
   and the arrow keys move within the value. Changing one digit of "250" meant
   retyping it, and even a double-click selected the whole value.

   A mouse click now goes straight to editing, with the caret where the click
   landed; a double-click selects the whole value to type over. The keyboard
   is untouched: a cell reached with Tab, Enter or the arrows is still selected,
   so typing replaces it and fast data entry works as before.

   Editing is switched on by sending the table the double-click it already
   understands, so the table's own state stays in charge of the cell.
   --------------------------------------------------------------------------- */
(function () {
  "use strict";

  var CELL_INPUT = "input.dash-cell-value";
  var MAX_FRAMES = 12;
  var canvas = null;
  // Bumped by every click and double-click, so a caret placement still
  // waiting on the table cannot undo a newer double-click's selection.
  var generation = 0;

  function textWidth(text, font) {
    canvas = canvas || document.createElement("canvas");
    var ctx = canvas.getContext("2d");
    ctx.font = font;
    return ctx.measureText(text).width;
  }

  // Index in the input's value nearest to a click at clientX.
  function caretIndexAt(input, clientX) {
    var value = input.value || "";
    if (!value) {
      return 0;
    }
    var style = getComputedStyle(input);
    var rect = input.getBoundingClientRect();
    var left = rect.left + parseFloat(style.borderLeftWidth) + parseFloat(style.paddingLeft);
    var width = input.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    var full = textWidth(value, style.font);

    var start = left;
    if (style.textAlign === "right" || style.textAlign === "end") {
      start = left + width - full;
    } else if (style.textAlign === "center") {
      start = left + (width - full) / 2;
    }

    var best = value.length;
    var bestDistance = Infinity;
    for (var i = 0; i <= value.length; i += 1) {
      var distance = Math.abs(start + textWidth(value.slice(0, i), style.font) - clientX);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = i;
      }
    }
    return best;
  }

  function placeCaret(td, clientX, frames, token) {
    if (token !== generation) {
      return;
    }
    var input = td.isConnected ? td.querySelector(CELL_INPUT) : null;
    if (input && input.classList.contains("focused")) {
      var index = caretIndexAt(input, clientX);
      input.focus();
      input.setSelectionRange(index, index);
      return;
    }
    if (frames > 0) {
      window.requestAnimationFrame(function () {
        placeCaret(td, clientX, frames - 1, token);
      });
    }
  }

  function startEditing(td, event, frames, token) {
    if (token !== generation) {
      return;
    }
    var input = td.isConnected ? td.querySelector(CELL_INPUT) : null;
    if (!input) {
      // The table renders the cell's input a frame or two after the click.
      if (frames > 0) {
        window.requestAnimationFrame(function () {
          startEditing(td, event, frames - 1, token);
        });
      }
      return;
    }
    if (input.classList.contains("unfocused")) {
      td.dispatchEvent(
        new MouseEvent("dblclick", {
          bubbles: true,
          cancelable: true,
          clientX: event.clientX,
          clientY: event.clientY,
        })
      );
    }
    placeCaret(td, event.clientX, MAX_FRAMES, token);
  }

  function selectValue(td, frames, token) {
    if (token !== generation) {
      return;
    }
    var input = td.isConnected ? td.querySelector(CELL_INPUT) : null;
    if (input && input.classList.contains("focused")) {
      input.focus();
      input.select();
      return;
    }
    if (frames > 0) {
      window.requestAnimationFrame(function () {
        selectValue(td, frames - 1, token);
      });
    }
  }

  // Dash 4's table saves an edit when the input loses focus, but a click on
  // another cell moves the selection first and the edit is dropped -- only Tab
  // and Enter kept a value. Every click now starts an edit, so every value
  // typed and then clicked away from vanished. Blur the cell being edited
  // before the table sees the mouse, while it still counts as the active cell.
  document.addEventListener(
    "mousedown",
    function (event) {
      var editing = document.activeElement;
      if (
        editing &&
        editing.matches &&
        editing.matches(".dash-spreadsheet " + CELL_INPUT + ".focused") &&
        !editing.contains(event.target)
      ) {
        editing.blur();
      }
    },
    true
  );

  document.addEventListener("click", function (event) {
    // Only a plain single click: leave shift-click range selection and the
    // second click of a double-click (which selects the value) alone.
    if (event.button !== 0 || event.detail > 1 || event.shiftKey || event.metaKey || event.ctrlKey) {
      return;
    }
    var td = event.target.closest && event.target.closest(".dash-spreadsheet td.dash-cell");
    if (!td) {
      return;
    }
    generation += 1;
    startEditing(td, event, MAX_FRAMES, generation);
  });

  // A real double-click selects the whole value, ready to be typed over.
  document.addEventListener("dblclick", function (event) {
    if (!event.isTrusted) {
      return; // the one this script sends to start editing
    }
    var td = event.target.closest && event.target.closest(".dash-spreadsheet td.dash-cell");
    if (!td) {
      return;
    }
    generation += 1;
    selectValue(td, MAX_FRAMES, generation);
  });
})();
