import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { DagDetailComponent } from './dag-detail.component';

describe('DagDetailComponent', () => {
  let component: DagDetailComponent;
  let fixture: ComponentFixture<DagDetailComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ DagDetailComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(DagDetailComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
