import React from 'react';

export default function BrandingImage() {
  const [hasSvg, setHasSvg] = React.useState<boolean | null>(null);

  React.useEffect(() => {
    let mounted = true;
    // Prefer the user-provided SVG filename, then traced, then generic branding.svg
    Promise.all([
      fetch('/image-removebg-preview.svg', { method: 'HEAD' }).then(r => ({ key: 'provided', ok: r.ok })).catch(() => ({ key: 'provided', ok: false })),
      fetch('/branding-traced.svg', { method: 'HEAD' }).then(r => ({ key: 'traced', ok: r.ok })).catch(() => ({ key: 'traced', ok: false })),
      fetch('/branding.svg', { method: 'HEAD' }).then(r => ({ key: 'branding', ok: r.ok })).catch(() => ({ key: 'branding', ok: false })),
    ]).then(results => {
      if (!mounted) return;
      const found = results.find(r => r.ok);
      setHasSvg(found ? true : false);
    }).catch(() => { if (mounted) setHasSvg(false); });
    return () => { mounted = false; };
  }, []);

  // If an SVG wrapper exists in public, prefer that (it can embed a PNG or be a real SVG).
  if (hasSvg) {
    // choose the best available file in order
    const candidates = ['/image-removebg-preview.svg', '/branding-traced.svg', '/branding.svg'];
    const chosen = candidates[0];
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', paddingBottom: 8, flexShrink: 0 }}>
        <img src={chosen} alt="branding" style={{ width: '100%', maxHeight: 96, height: 'auto', objectFit: 'contain' }} />
      </div>
    );
  }

  // Fallback: simple placeholder box (previous inline path was truncated).
  return (
    <div style={{ width: '100%', height: 72, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.02)', borderRadius: 8, color: '#94a3b8', fontSize: 13, flexShrink: 0 }}>
      No branding set
    </div>
  );
}
