import { TestBed } from '@angular/core/testing';

import { DynamicresourceService } from './dynamicresource.service';

describe('DynamicresourceService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: DynamicresourceService = TestBed.get(DynamicresourceService);
    expect(service).toBeTruthy();
  });
});
