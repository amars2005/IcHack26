import React from 'react';

export default function BrandingImage() {
<<<<<<< HEAD
  const [svgHtml, setSvgHtml] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let mounted = true;
    const candidates = ['/BrandingImage.svg', '/image-removebg-preview.svg', '/branding-traced.svg', '/branding.svg'];
    (async () => {
      try {
        for (const c of candidates) {
          try {
            const res = await fetch(c);
            if (!mounted) return;
            if (!res.ok) continue;
            const text = await res.text();
            // Parse the SVG and remove any full-coverage dark/background paths (e.g., fill="#0A0A0A" or white backgrounds)
            try {
              const parser = new DOMParser();
              const doc = parser.parseFromString(text, 'image/svg+xml');
              const paths = Array.from(doc.querySelectorAll('path'));
              for (const p of paths) {
                const fill = (p.getAttribute('fill') || '').trim();
                // remove obvious solid background rectangles (very dark or very light fills)
                if (/^(#?0a0a0a|#?000000|black)$/i.test(fill) || /^(#?ffffff|#?fff|white)$/i.test(fill)) {
                  p.remove();
                }
              }
              const serializer = new XMLSerializer();
              const cleaned = serializer.serializeToString(doc.documentElement);
              setSvgHtml(cleaned);
              setLoading(false);
              return;
            } catch (err) {
              // parsing failed — fallback to raw image src
              setSvgHtml(null);
              setLoading(false);
              return;
            }
          } catch (e) {
            // try next candidate
          }
        }
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, []);

  if (loading) return <div style={{ height: 72, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>Loading…</div>;

  if (svgHtml) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', paddingBottom: 8, flexShrink: 0 }} dangerouslySetInnerHTML={{ __html: svgHtml }} />
    );
  }

=======
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
>>>>>>> main
  return (
    <div style={{ width: '100%', height: 72, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(255,255,255,0.02)', borderRadius: 8, color: '#94a3b8', fontSize: 13, flexShrink: 0 }}>
      No branding set
    </div>
  );
}
