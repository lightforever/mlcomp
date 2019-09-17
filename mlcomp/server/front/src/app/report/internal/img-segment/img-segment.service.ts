import { Injectable } from '@angular/core';
import {BaseService} from "../../../base.service";

@Injectable({
  providedIn: 'root'
})
export class ImgSegmentService extends BaseService{
   protected collection_part: string = 'img_segment';
   protected single_part: string;

}
