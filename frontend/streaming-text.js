/**
 * Whole-text crossfade helper for streaming transcripts.
 *
 * Both the realtime ASR and TS-ASR pages call ``AmphionStreamingText.apply``
 * on every partial / final update. The previous version did a per-character
 * diff so only changed glyphs animated; this version takes a much simpler
 * approach: each new text replaces the previous one as a single block,
 * with the outgoing string fading out and the incoming string fading in
 * on top of it (CSS Grid stacks both layers in the same cell so the
 * paragraph height grows / shrinks once per swap rather than per char).
 *
 * The crossfade reads as a calm "the bubble updated to a new line of
 * text" instead of the busy left-to-right wave the diff version produced,
 * which the team found visually noisy for ASR partials that frequently
 * rewrite multiple words at once.
 *
 * DOM convention (mounted into ``.bubble-text``):
 *
 *   <p class="bubble-text">
 *     <span class="text-frame is-current">latest text</span>
 *     <!-- briefly during a swap: -->
 *     <span class="text-frame is-leaving">previous text</span>
 *   </p>
 *
 * Public API:
 *   AmphionStreamingText.apply(targetEl, newText)
 *     Crossfade the bubble's text from whatever is currently shown to
 *     ``newText``. No-ops when the new text matches the current text.
 *     Safe to call on a target that currently holds plain text — that
 *     path skips the fade-out and just installs the first layer.
 *
 *   AmphionStreamingText.reset(targetEl)
 *     Wipe the target's children synchronously. Useful when discarding
 *     a bubble or re-using the same DOM node for a fresh utterance.
 */

(function () {
  'use strict';

  // Fallback timeout for the leaving layer's removal: covers the CSS
  // fade-out duration with some headroom. Used if animationend never
  // fires (reduced motion, hidden tab, animation cancelled) so removed
  // frames don't accumulate in the DOM forever.
  const REMOVAL_FALLBACK_MS = 500;

  function removeNode(node) {
    if (node && node.parentNode) node.parentNode.removeChild(node);
  }

  function apply(targetEl, newText) {
    if (!targetEl) return;
    const newStr = String(newText == null ? '' : newText);

    // Find the layer that is currently fully visible (or finishing its
    // fade-in). It's the only thing the user "sees" right now.
    const current = targetEl.querySelector(':scope > .text-frame.is-current');
    const currentText = current ? current.textContent : '';

    // No-op when nothing actually changed — keeps us from restarting the
    // fade-in animation on duplicate partials (e.g. server retransmits).
    if (newStr === currentText && (current || newStr === '')) return;

    // If a fast burst of updates already left a previous frame mid-fade,
    // yank it out instantly so we never stack more than two layers (one
    // leaving, one current). This keeps the DOM tidy under bad cases like
    // the user dragging the page while partials are streaming.
    targetEl
      .querySelectorAll(':scope > .text-frame.is-leaving')
      .forEach(removeNode);

    if (current) {
      // Demote the visible layer to "leaving" so the CSS keyframe takes
      // over and fades it out. Once the animation finishes (or our
      // fallback timer fires) the node is removed from the DOM.
      current.classList.remove('is-current');
      current.classList.add('is-leaving');
      let cleaned = false;
      const cleanup = () => {
        if (cleaned) return;
        cleaned = true;
        current.removeEventListener('animationend', cleanup);
        removeNode(current);
      };
      current.addEventListener('animationend', cleanup);
      setTimeout(cleanup, REMOVAL_FALLBACK_MS);
    }

    // Empty new text just means "fade out whatever's there"; we don't
    // mount an empty span (it would still consume a grid cell and could
    // throw off the line-height baseline of an empty bubble).
    if (newStr.length === 0) return;

    const next = document.createElement('span');
    next.className = 'text-frame is-current';
    next.textContent = newStr;
    targetEl.appendChild(next);
  }

  function reset(targetEl) {
    if (!targetEl) return;
    targetEl.textContent = '';
  }

  window.AmphionStreamingText = { apply, reset };
})();
