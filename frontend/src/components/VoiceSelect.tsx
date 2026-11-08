import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { Voice } from '../types';

type State = 'idle' | 'busy' | 'error';

/** Picks which ElevenLabs voice the backend speaks phrases in. */
export default function VoiceSelect() {
  const [voices, setVoices] = useState<Voice[]>([]);
  const [selected, setSelected] = useState('');
  const [state, setState] = useState<State>('idle');

  useEffect(() => {
    api
      .listVoices()
      .then((v) => {
        setVoices(v);
        if (v.length) setSelected(v[0].voice_id);
      })
      .catch(() => setState('error'));
  }, []);

  const onChange = (voiceId: string) => {
    setSelected(voiceId);
    setState('busy');
    api
      .selectVoice(voiceId)
      .then(() => setState('idle'))
      .catch(() => setState('error'));
  };

  if (state === 'error' && !voices.length) {
    return <span className="text-sm text-muted">Voice unavailable</span>;
  }
  if (!voices.length) return null;

  return (
    <select
      className="btn !min-h-[2rem] !px-3 !py-0 !text-sm"
      value={selected}
      onChange={(e) => onChange(e.target.value)}
      disabled={state === 'busy'}
      aria-label="Voice"
    >
      {voices.map((v) => (
        <option key={v.voice_id} value={v.voice_id}>
          {v.name}
        </option>
      ))}
    </select>
  );
}
