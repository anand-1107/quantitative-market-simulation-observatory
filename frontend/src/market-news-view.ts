import type { HistoricalNewsArticle, MarketEventNews } from './types'

export type NewsCard = {
  article: HistoricalNewsArticle
  rank: number | null
  score: number | null
  signals: Record<string, number | string> | null
  eventId: string | null
  eventLabel: string | null
  eventTimestamp: string | null
}

export const isLatestRequest = (requestId: number, latestRequestId: number): boolean => requestId === latestRequestId

/** Keep the provider response visible even when an article cannot be ranked to an event. */
export function buildNewsCards(
  news: MarketEventNews,
  selectedEventId: string | undefined,
  events: Array<{ event_id: string; category: string; timestamp: string }>,
): NewsCard[] {
  const rankingByUrl = new Map<string, Omit<NewsCard, 'article'>>()
  const eventById = new Map(events.map((event) => [event.event_id, event]))
  const matches = [...news.event_matches].sort((a, b) =>
    Number(b.event_id === selectedEventId) - Number(a.event_id === selectedEventId),
  )
  for (const match of matches) {
    const event = eventById.get(match.event_id)
    for (const ranked of match.ranked_articles) {
      const url = ranked.article.original_url
      if (!rankingByUrl.has(url)) rankingByUrl.set(url, {
        rank: Number.isFinite(ranked.rank) ? ranked.rank : null,
        score: typeof ranked.score === 'number' && Number.isFinite(ranked.score) ? ranked.score : null,
        signals: ranked.signals ?? null, eventId: match.event_id,
        eventLabel: event?.category ?? match.category, eventTimestamp: event?.timestamp ?? null,
      })
    }
  }
  // The provider's article list is the source of truth. Do not filter or deduplicate it
  // based on event matches, missing ranking scores, or duplicate source URLs.
  return news.articles.map((article) => ({
    article,
    ...(rankingByUrl.get(article.original_url) ?? {
      rank: null, score: null, signals: null, eventId: null, eventLabel: null, eventTimestamp: null,
    }),
  })).sort((a, b) =>
    Number(b.eventId === selectedEventId) - Number(a.eventId === selectedEventId)
      || (a.rank ?? Number.MAX_SAFE_INTEGER) - (b.rank ?? Number.MAX_SAFE_INTEGER),
  )
}

export function newsSearchStatus(news: MarketEventNews | null, error: string, busy: boolean):
  'loading' | 'error' | 'not-searched' | 'unconfigured' | 'empty' | 'results' | 'unranked' | null {
  if (busy) return 'loading'
  if (error) return 'error'
  if (!news) return null
  if (!news.searched && !news.configured) return 'unconfigured'
  if (!news.searched) return 'not-searched'
  if (news.articles.length === 0) return 'empty'
  if (!news.event_matches.some((match) => match.ranked_articles.length > 0)) return 'unranked'
  return 'results'
}
