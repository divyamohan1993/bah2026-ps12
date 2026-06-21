import { Globe2 } from 'lucide-react';
import type { SceneIndexEntry } from '@/types/manifest';
import { Select } from '@/components/ui/Select';

interface SceneSelectorProps {
  scenes: SceneIndexEntry[];
  value: string;
  onChange: (sceneId: string) => void;
}

/** Dropdown to switch between available scenes. */
export function SceneSelector({ scenes, value, onChange }: SceneSelectorProps) {
  return (
    <div className="flex items-center gap-2">
      <Globe2 className="h-4 w-4 text-cyan/70" />
      <Select
        aria-label="Select scene"
        value={value}
        onValueChange={onChange}
        options={scenes.map((s) => ({ value: s.scene_id, label: s.title }))}
        className="min-w-[220px]"
      />
    </div>
  );
}
