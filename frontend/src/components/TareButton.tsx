import { useState } from 'react';
import { api } from '../api/client';

type State = 'idle' | 'busy' | 'done' | 'error';

/** Hold the jaw and head still in a neutral pose while this runs (~1.5 s). Do it once per session, after the hat is on. */
export default function TareButton() {
  const [state, setState] = useState<State>('idle');
  const run = () => {
    setState('busy');
    api
      .tare()
      .then(() => setState('done'))
      .catch(() => setState('error'));
  };
  const label = { idle: 'Tare', busy: 'Hold still…', done: 'Tared ✓', error: 'Tare failed' }[state];
  return (
    <button className={`btn !min-h-[2rem] !px-3 !py-0 !text-sm ${state === 'done' ? 'btn-active' : ''}`} onClick={run} disabled={state === 'busy'}>
      {label}
    </button>
  );
}
