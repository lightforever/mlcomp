import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { DagsComponent } from './dags.component';

describe('DagComponent', () => {
  let component: DagsComponent;
  let fixture: ComponentFixture<DagsComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ DagsComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(DagsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
