import React from 'react';
import Select from 'react-select';

const SearchFormHTML = ({
  targetItem,
  algorithm,
  startIndex,
  endIndex,
  result,
  error,
  handleSubmit,
  setTargetItem,
  setAlgorithm,
  setStartIndex,
  setEndIndex,
  algorithmOptions
}) => (
  <div>
    <form onSubmit={handleSubmit} className="form">
      <label htmlFor="target">Target Item</label>
      <input
        id="target"
        type="text"
        placeholder="Enter item name"
        value={targetItem}
        onChange={e => setTargetItem(e.target.value)}
        required
      />

      <label htmlFor="algorithm">Choose Algorithm</label>
      <Select
        id="algorithm"
        options={algorithmOptions}
        onChange={setAlgorithm}
        value={algorithm}
        placeholder="Select an algorithm..."
        isSearchable
      />

      <label htmlFor="start">Start Index (optional)</label>
      <input
        id="start"
        type="number"
        placeholder="Start index"
        value={startIndex}
        onChange={e => setStartIndex(e.target.value)}
      />

      <label htmlFor="end">End Index (optional)</label>
      <input
        id="end"
        type="number"
        placeholder="End index"
        value={endIndex}
        onChange={e => setEndIndex(e.target.value)}
      />

      <button type="submit">Search</button>
    </form>

    {error && <p className="error">❌ {error}</p>}

    {result && (
      <div className="result">
        <h2>Results (via {result.algorithm_used})</h2>
        <pre>{JSON.stringify(result.results, null, 2)}</pre>
      </div>
    )}
  </div>
);

export default SearchFormHTML;