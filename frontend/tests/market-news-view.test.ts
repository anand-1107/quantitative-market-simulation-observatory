import assert from 'node:assert/strict'
import test from 'node:test'
import { buildNewsCards, isLatestRequest, newsSearchStatus } from '../src/market-news-view.ts'
import type { MarketEventNews } from '../src/types.ts'

// These records are synthetic test fixtures and are never used as application news.
const fixture = (overrides: Partial<MarketEventNews> = {}): MarketEventNews => ({
  provider: 'Finnhub company news', configured: true, searched: true, cached: false,
  retrieved_at: '2024-01-03T00:00:00Z',
  articles: [
    { headline: 'Test fixture: earnings report', publisher: 'Fixture Publisher', original_url: 'https://fixture.invalid/article', publication_timestamp: '2024-01-02T12:00:00Z', publication_timezone: 'UTC', retrieved_timestamp: '2024-01-03T00:00:00Z', relevant_instrument: 'TEST', provider_query_ticker: 'TEST', excerpt: null, category: 'company', source_provider: 'fixture' },
    { headline: 'Test fixture: undated item', publisher: 'Fixture Publisher', original_url: 'https://fixture.invalid/undated', publication_timestamp: null, publication_timezone: null, retrieved_timestamp: '2024-01-03T00:00:00Z', relevant_instrument: null, provider_query_ticker: 'TEST', excerpt: null, category: 'general', source_provider: 'fixture' },
  ],
  event_matches: [{ event_id: 'evt1', category: 'large_absolute_return', ranked_articles: [{ rank: 1, score: 0.87, article: { headline: 'Test fixture: earnings report', publisher: 'Fixture Publisher', original_url: 'https://fixture.invalid/article', publication_timestamp: '2024-01-02T12:00:00Z', publication_timezone: 'UTC', retrieved_timestamp: '2024-01-03T00:00:00Z', relevant_instrument: 'TEST', provider_query_ticker: 'TEST', excerpt: null, category: 'company', source_provider: 'fixture' }, signals: { temporal_proximity_score: .9, explanation: 'Fixture ranking explanation' } }] }],
  ...overrides,
})

test('all returned articles remain visible and ranked matches include event provenance', () => {
  const cards = buildNewsCards(fixture(), 'evt1', [{ event_id: 'evt1', category: 'large_absolute_return', timestamp: '2024-01-02' }])
  assert.equal(cards.length, 2)
  assert.equal(cards[0].score, .87)
  assert.equal(cards[0].eventTimestamp, '2024-01-02')
  assert.equal(cards[1].rank, null)
  assert.equal(cards[1].eventId, null)
})

test('populated response articles render independently of an empty event_matches array', () => {
  const news = fixture({ event_matches: [] })
  const cards = buildNewsCards(news, undefined, [])
  assert.equal(cards.length, news.articles.length)
  assert.deepEqual(cards.map((card) => card.article.headline), news.articles.map((article) => article.headline))
  assert.equal(cards[0].article.original_url, 'https://fixture.invalid/article')
  assert.equal(cards[0].article.publisher, 'Fixture Publisher')
  assert.equal(cards[0].article.publication_timestamp, '2024-01-02T12:00:00Z')
  assert.equal(cards[0].article.relevant_instrument, 'TEST')
  assert.equal(cards[0].article.excerpt, null)
  assert.equal(cards[0].eventId, null)
  assert.equal(cards[0].score, null)
  assert.equal(cards[1].article.publication_timestamp, null)
  assert.equal(cards[1].article.relevant_instrument, null)
})

test('provider article order and duplicate URLs are preserved rather than filtered by ranking', () => {
  const base = fixture({ event_matches: [] })
  const duplicate = { ...base.articles[0], headline: 'Test fixture: duplicate URL record' }
  const news = fixture({ articles: [...base.articles, duplicate], event_matches: [] })
  const cards = buildNewsCards(news, undefined, [])
  assert.equal(cards.length, 3)
  assert.equal(cards[0].article.headline, base.articles[0].headline)
  assert.equal(cards[2].article.headline, duplicate.headline)
})

test('news states distinguish not searched, unconfigured, empty, results and errors', () => {
  assert.equal(newsSearchStatus(null, '', false), null)
  assert.equal(newsSearchStatus(null, '', true), 'loading')
  assert.equal(newsSearchStatus(null, 'provider unavailable', false), 'error')
  assert.equal(newsSearchStatus(fixture({ configured: false, searched: false, articles: [], event_matches: [] }), '', false), 'unconfigured')
  assert.equal(newsSearchStatus(fixture({ articles: [], event_matches: [] }), '', false), 'empty')
  assert.equal(newsSearchStatus(fixture({ event_matches: [] }), '', false), 'unranked')
  assert.equal(newsSearchStatus(fixture(), '', false), 'results')
})

test('a stale asynchronous response cannot replace the latest search result', () => {
  const latestRequestId = 4
  assert.equal(isLatestRequest(3, latestRequestId), false)
  assert.equal(isLatestRequest(4, latestRequestId), true)
})
