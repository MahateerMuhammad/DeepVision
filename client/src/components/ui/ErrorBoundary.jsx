import { Component } from "react";

/**
 * Catches render/runtime errors in its subtree (e.g. a WebGL canvas losing its
 * context) and shows a recoverable fallback instead of white-screening the page.
 * Pass a changing `resetKey` (e.g. the network id) to auto-clear the error when
 * upstream state changes.
 */
export default class ErrorBoundary extends Component {
  state = { error: null, prevKey: this.props.resetKey };

  static getDerivedStateFromError(error) {
    return { error };
  }

  static getDerivedStateFromProps(props, state) {
    // Clear a prior error automatically when the upstream reset key changes
    // (e.g. the network was re-forged), without setState-in-didUpdate.
    if (props.resetKey !== state.prevKey) {
      return { error: null, prevKey: props.resetKey };
    }
    return null;
  }

  reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      return (
        <div className="dot-grid flex h-full items-center justify-center">
          <div className="border border-crimson bg-panel px-6 py-4 text-center shadow-instrument" style={{ borderRadius: 4 }}>
            <p className="micro-label !text-crimson mb-1">Render fault</p>
            <p className="mb-3 max-w-xs text-xs text-ink-soft">
              The view hit an unexpected error. Reset to recover, or re-forge the network.
            </p>
            <button
              type="button"
              onClick={this.reset}
              className="h-8 border border-ink bg-ink px-3 text-[11px] font-semibold uppercase tracking-[0.08em] text-white active:translate-y-px"
              style={{ borderRadius: 3 }}
            >
              Reset view
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
